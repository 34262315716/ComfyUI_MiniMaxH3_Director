"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging
import weakref

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def _evict_dead_loaded_models() -> int:
    """Pop Comfy LoadedModel slots that ``free_memory`` will skip forever.

    ``is_dead()`` means the ModelPatcher weakref is gone while the shared
    MiniMaxH3 module is still alive (graph MODEL / Sage cycle). Those slots
    log ``Potential memory leak detected with model MiniMaxH3`` and then sit
    in ``current_loaded_models``, so later unloads cannot touch them.
    Evicting the slot does not copy weights; it restores unload bookkeeping.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            name = "?"
            try:
                real = cur.real_model()
                name = type(real).__name__ if real is not None else "?"
            except Exception:
                pass
            models.pop(i)
            evicted += 1
            log.info("MiniMax H3 Director: evicted dead LoadedModel slot (%s)", name)
        except Exception:
            continue
    return evicted


def _loaded_slot_for(patcher):
    """把 ModelPatcher 映射到 ``current_loaded_models`` 里的那个 LoadedModel。

    必须返回**同一个对象**，不能传 ModelPatcher 充数 —— Comfy 的
    ``LoadedModel.__eq__`` 是 ``self.model is other.model``，拿 ModelPatcher
    去比会去读它的 ``.model`` 属性（那是内层 BaseModel），永远比不中。

    LoRA patch 会把模型变成**克隆**（对象不同但 ``clone_base_uuid`` 相同），
    所以除了 ``is`` 还要按 uuid 兜一层 —— 一条链上挂了 LoRA 时，
    ``current_loaded_models`` 里存的是克隆，只靠 ``is`` 是匹配不到的。
    """
    if patcher is None:
        return None
    try:
        import comfy.model_management as mm
    except Exception:
        return None
    base_uuid = getattr(patcher, "clone_base_uuid", None)
    for slot in list(getattr(mm, "current_loaded_models", ()) or ()):
        try:
            inner = getattr(slot, "model", None)
            if inner is patcher:
                return slot
            if base_uuid is not None and getattr(inner, "clone_base_uuid", None) == base_uuid:
                return slot
        except Exception:
            continue
    return None


def unload_except(keep_patchers) -> int:
    """卸载所有已加载模型，**除了** ``keep_patchers`` 里的那些。返回卸掉几个。

    为什么要有这个：``mm.unload_all_models()`` 在段与段之间会把**下一个阶段
    正要用的模型**也一起赶走。内存充裕时无所谓；内存紧张时，被赶走的权重会
    落到页面文件，下次装载要再从硬盘读回来 —— 而这一趟远比直接读模型文件慢。

    传空列表时行为与 ``unload_all_models()`` **完全一致**，所以是向后兼容的。
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0

    keep = []
    for patcher in keep_patchers or ():
        slot = _loaded_slot_for(patcher)
        if slot is not None and slot not in keep:
            keep.append(slot)

    try:
        devices = list(mm.get_all_torch_devices())
    except Exception:
        try:
            devices = [mm.get_torch_device()]
        except Exception:
            return 0

    freed = 0
    for device in devices:
        try:
            freed += len(mm.free_memory(1e30, device, keep_loaded=keep))
        except Exception as exc:
            # 选择性卸载失败不算致命：退回全卸，最多慢一点，不能让它把流程打断
            log.warning("selective unload failed on %s (%s); falling back to unload_all", device, exc)
            try:
                mm.unload_all_models()
            except Exception:
                pass
            return freed
    return freed



# ---------------------------------------------------------------- 阶段调度

#: 每个阶段**真正会用到**的模型角色。
#:
#: 这张表来自代码在各阶段的实际行为，与机器配置无关 —— 换任何模型组合都成立：
#:
#:   context_encode  文本编码器编提示词；视频 VAE 编参考图；音频 VAE 编参考音频
#:   sample          扩散模型采样；二采模型 / 放大网做 refine；VAE 出画中画预览
#:   decode          视频 VAE + 音频 VAE
#:
#: **表里没有的角色，就是该阶段可以安全卸掉的。** 至于卸了值不值（读盘 vs 腾内存），
#: 由 should_evict() 按物理内存自动判断，不在这里写死。
PHASE_MODEL_ROLES = {
    "context_encode": ("clip", "vae", "audio_vae"),
    "sample": ("model", "refine_model", "upscale_model", "vae"),
    "decode": ("vae", "audio_vae"),
}

#: 已加载模型的总量超过物理内存这个比例时，才值得卸载。
#: 低于它说明内存本来就够，卸载只会白白多读几次盘。
EVICT_RAM_RATIO = 0.65


def phase_keep_pool(phase, pool):
    """该阶段要保留的模型列表。``pool`` 是 {角色名: ModelPatcher}。"""
    return [pool[r] for r in PHASE_MODEL_ROLES.get(phase, ()) if pool.get(r) is not None]


def _loaded_model_bytes():
    """当前已加载模型的总体积（按 model_memory 算，含 offload 部分）。"""
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    total = 0
    for slot in list(getattr(mm, "current_loaded_models", ()) or ()):
        try:
            total += int(slot.model_memory())
        except Exception:
            continue
    return total


def _ram_total():
    try:
        import torch
        import comfy.model_management as mm
        return int(mm.get_total_memory(torch.device("cpu")))
    except Exception:
        return 0


def _patcher_bytes(patcher) -> int:
    try:
        return int(patcher.model_size())
    except Exception:
        return 0


def should_evict(pool=None) -> bool:
    """值不值得按阶段卸？内存宽裕时一个都不卸。

    ⚠ 判据是**整个工作流要用的模型总量**，不是"当前已加载了多少"。

    为什么不能用当前加载量判断：某个阶段结束时场上可能只剩下编码器和 VAE，
    看着很宽裕；但下一个阶段要装的生成模型还没算进去，一装就会溢出 ——
    用当前值判会得出"不用卸"，然后一装就爆。
    模型总量是工作流的固有属性，不随加载状态变化，所以它才是对的判据。

    拿不到内存数据时按保守来（卸）。
    """
    ram = _ram_total()
    if ram <= 0:
        return True
    total = 0
    for patcher in (pool or {}).values():
        if patcher is not None:
            total += _patcher_bytes(patcher)
    if total <= 0:
        total = _loaded_model_bytes()
    if total <= 0:
        return False
    return total > ram * EVICT_RAM_RATIO



#: 所有还活着的 ``ModelVBAR`` 的弱引用。
#:
#: 为什么需要它：从 ``_model_pool`` 出发枚举 VBAR 是**不可靠**的 ——
#: ``pool["clip"]`` 是 ``CLIP`` 包装对象、``pool["vae"]`` 是 ``VAE``，
#: 它们不一定能走到真正的 ``ModelPatcher.model.dynamic_vbars``。实测出现过
#: 「枚举到的每个 vbar 都报 ``loaded=0.00G``，而 ``get_total_vram_usage()``
#: 仍报 4.02G」—— 那 4G 就挂在枚举不到的**孤儿 VBAR** 上，而它正是段 2
#: 拿不到显存的原因。
#:
#: 所以在 ``ModelVBAR.__init__`` 上挂一个记录器，凡是创建过的都留个弱引用。
#: 弱引用不会阻止回收，也不改变任何行为。
_VBAR_REGISTRY = []
_VBAR_HOOKED = False


def _hook_vbar_registry() -> bool:
    """给 ``ModelVBAR.__init__`` 挂记录器（幂等）。"""
    global _VBAR_HOOKED
    if _VBAR_HOOKED:
        return True
    try:
        import comfy_aimdo.model_vbar as _mvbar
    except Exception:
        return False
    cls = getattr(_mvbar, "ModelVBAR", None)
    if cls is None:
        return False
    try:
        orig = cls.__init__

        def _init(self, *args, **kwargs):
            orig(self, *args, **kwargs)
            try:
                _VBAR_REGISTRY.append(weakref.ref(self))
            except Exception:
                pass

        cls.__init__ = _init
        _VBAR_HOOKED = True
        return True
    except Exception as exc:
        log.debug("vbar registry hook skipped: %s", exc)
        return False


def _live_vbars():
    """注册表里所有还活着的 VBAR。"""
    out = []
    try:
        alive = []
        for ref in _VBAR_REGISTRY:
            obj = ref()
            if obj is not None:
                alive.append(ref)
                out.append(obj)
        _VBAR_REGISTRY[:] = alive          # 顺手清掉死掉的
    except Exception:
        pass
    return out


def _collect_vbars(mm, pool=None):
    """收集所有**还在场上**的动态模型 VBAR 句柄。

    为什么不能只查 ``current_loaded_models``：这是第一版补丁失败的原因。
    ``unload_all_models()`` 只把模型从 ComfyUI 的列表里摘掉，**对象本身仍被
    导演台的 ``_model_pool`` 握着**（``executor_core.py`` 的
    ``{"model": model, "clip": clip, ...}`` 在整整 5 段的循环里都活着 ——
    而且节点入参本身就是局部变量，这个引用躲不掉）。于是：

        ModelPatcher 活着 → ModelVBAR 活着 → ``__del__`` 不触发
        → ``vbar_free()`` 永不调用 → 那 4~4.7 GB 显存永远不还

    实测吻合：``models == []`` 而 ``aimdo_vram`` 仍有 4.68~5.54 GB。
    ``reset_cast_buffers()`` / ``vbars_reset_watermark_limits()`` 都**不碰
    VBAR 的生命周期**，所以对它们无效 —— 必须显式 ``free_memory()`` 放页。

    所以这里**以 ``pool`` 为主**（它才是引用真正所在），已加载列表只作兜底。
    """
    out = []
    seen = set()

    def _take(patcher, vbar):
        if vbar is None or id(vbar) in seen:
            return
        seen.add(id(vbar))
        out.append((patcher, vbar))

    def _from_patcher(patcher):
        if patcher is None:
            return
        # 包装对象（CLIP / VAE）真正的 patcher 在 .patcher 上
        for cand in (patcher, getattr(patcher, "patcher", None)):
            if cand is None:
                continue
            mdl = getattr(cand, "model", None)
            for _dev, vbar in (getattr(mdl, "dynamic_vbars", {}) or {}).items():
                _take(cand, vbar)

    try:
        for patcher in (pool or {}).values():
            _from_patcher(patcher)
    except Exception as exc:
        log.debug("collect vbars from pool skipped: %s", exc)

    try:
        for lm in list(getattr(mm, "current_loaded_models", []) or []):
            _from_patcher(getattr(lm, "model", None))
    except Exception as exc:
        log.debug("collect vbars from loaded skipped: %s", exc)

    # ★ 注册表兜底：凡是创建过的 VBAR 都在这里，不管它挂在哪。
    #   上面两条只能找到"还挂得上"的；实测有 VBAR 两条都够不着，
    #   而 get_total_vram_usage() 仍把它算进总数。
    _hook_vbar_registry()
    for vbar in _live_vbars():
        _take(None, vbar)

    return out


def _unpin_model(patcher):
    """解除该模型**全部模块**的 VBAR 页 pin，并拆掉 ``_prefetch`` 棘轮。

    返回 ``(解pin次数, 拆棘轮次数)``。

    为什么必须有这一步：aimdo 的 ``fault()`` 会给页加 pin，只有 ``unpin()``
    才允许回收。独立模拟（``tools/aimdo_sim.py``）实测：

        fault 2GB  → residency 总页255 / 驻留64 / 被pin64   ← 全部被钉
        free_memory(全部)      → 返回 0.000 GB              ← 一个字节都放不掉
        vbar_unpin(alloc)      → 驻留 0 / 被pin 0，显存立刻还回

    真机日志里 ``Page N pin_count=1`` 出现 95 次，正是这个状态。

    ``alloc`` 就是模块上的 ``m._v``（``model_patcher.py`` 里
    ``m._v = vbar.alloc(v_weight_size)`` 设的），``vbar_unpin(alloc)`` 直接吃它；
    预取路径还会用 ``m._v_block``（``model_prefetch.py``），所以两个都试。

    解除 pin 不会丢数据 —— 页还在，只是**允许被回收**。真被回收了，
    ``ops.py`` 下次用权重前会比对页驻留签名，自动重新 fault。安全。
    """
    mdl = getattr(patcher, "model", None)
    if mdl is None:
        return 0, 0
    try:
        import comfy_aimdo.model_vbar as _mvbar
    except Exception:
        return 0, 0
    n = 0
    n_ratchet = 0
    try:
        modules = list(mdl.modules())
    except Exception as exc:
        log.debug("walk modules skipped: %s", exc)
        return 0, 0
    for m in modules:
        for attr in ("_v", "_v_block"):
            alloc = getattr(m, attr, None)
            if alloc is None:
                continue
            try:
                _mvbar.vbar_unpin(alloc)
                n += 1
            except Exception:
                pass
        # ★ 拆掉 _prefetch 棘轮。
        #   comfy/ops.py 里 cast_modules_with_vbar 会无条件给模块打上 _prefetch；
        #   下一次 cast 时 prefetched=True → offload_stream 恒为 None →
        #   uncast_bias_weight 第一行就 return，**unpin 永不执行**；
        #   而删 _prefetch 的那行在 `if not prefetched:` 里面，也永不执行。
        #   于是页从第一次 fault 起就被 pin_count=1 钉死。
        #   编码器（Llama2_）没有 prefetch 队列，所以 cleanup_prefetch_queues()
        #   也扫不到它。不拆这个棘轮，解完 pin 下一层又会重新锁死。
        if getattr(m, "_prefetch", None) is not None:
            try:
                delattr(m, "_prefetch")
                n_ratchet += 1
            except Exception:
                pass
    return n, n_ratchet


def _detach_vbar(patcher, vbar):
    """强制把模型从 VBAR 上摘干净，让 ``ModelVBAR.__del__`` 触发 ``vbar_free()``。

    为什么必须有这一步：``_unpin_model`` 只解 pin、不动引用。而模块上的
    ``m._v`` 存的是 ``(vbar, base_addr+offset, size)`` —— **它本身就是一条
    对 VBAR 的强引用**。只要任何一个模块还留着 ``_v``（或 ``_v_block``），
    那个 VBAR 的引用计数就永远不归零：

        ModelPatcher 活着 → dynamic_vbars 握着 VBAR → __del__ 不触发
        → vbar_free() 永不调用 → 那 4~4.7 GB 显存永远不还

    而 ``_model_pool`` 在整条分段循环里一直握着 model 引用，这个引用躲不掉。
    所以只能主动把 ``_v`` 摘掉，再自己把 VBAR 从 ``dynamic_vbars`` 里摘除。

    摘掉之后下次要用会走 ``_vbar_get(create=True)`` 建一个**新的** VBAR，
    模块重新拿到全新的 ``_v`` —— 所以摘之前必须确认**一个残留都不能有**。

    返回 ``(清掉的 _v 数, 摘掉的 VBAR 数, 是否摘成功)``。
    """
    mdl = getattr(patcher, "model", None)
    if mdl is None:
        return 0, 0, False
    try:
        import comfy_aimdo.model_vbar as _mvbar
    except Exception:
        return 0, 0, False

    try:
        modules = list(mdl.modules())
    except Exception:
        return 0, 0, False

    cleared = 0
    leftover = 0
    for m in modules:
        # _prefetch 是预取棘轮；_v_block 与 _v 一样是 (vbar, addr, size)，同样持引用
        for attr in ("_prefetch", "_v_block"):
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass
        if hasattr(m, "_v"):
            try:
                _mvbar.vbar_unpin(m._v)
            except Exception:
                pass
            try:
                delattr(m, "_v")
                cleared += 1
            except Exception:
                pass
            if hasattr(m, "_v"):
                leftover += 1

    if leftover:
        # ★ 安全闸：漏一个 _v，下次新建的 VBAR 就是另一段地址空间，
        #   而残留的 _v 还指着旧地址 —— 会读写到错误显存。宁可这次不摘。
        log.warning(
            "H3Director[vram] detach aborted: %d module(s) still hold _v", leftover
        )
        return cleared, 0, False

    detached = 0
    try:
        dvs = getattr(mdl, "dynamic_vbars", None)
        if dvs:
            for dev, v in list(dvs.items()):
                if v is vbar:
                    del dvs[dev]
                    detached += 1
    except Exception as exc:
        log.debug("dynamic_vbars detach skipped: %s", exc)

    if detached:
        gc.collect()          # 让 __del__ 尽早跑，vbar_free() 才会被执行
    return cleared, detached, detached > 0


def _release_aimdo_vram(mm, vbars=()) -> None:
    """释放 aimdo 三层显存预留。**段间最重要的一步。**

    涉及的**三层**结构（缺一不可）：

    1. ``VRAMBuffer``（cast buffer）—— 上限 ``DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE``
       （16 GB），且只增不减（``get()`` 只调 ``vrambuf_grow``，没有收缩接口）。
       释放靠 ``STREAM_AIMDO_CAST_BUFFERS.clear()`` 让引用计数归零、触发 ``__del__``。

    2. ``ModelVBAR`` 的**水位**（每模型的虚拟 BAR，按 ``model_size() * 10`` 创建，
       见 comfy/model_patcher.py）—— 持有物理驻留页，由水位限制控制驻留量。
       释放靠 ``vbars_reset_watermark_limits()``。

    3. ``ModelVBAR`` 的**优先级** —— 见下方注释，这是原先漏掉的一层。

    前两层都在 ``execution.py`` 节点执行的 finally 块里重置 —— 也就是说
    **只在「每个节点执行完之后」**。导演台是**一个**节点、内部循环所有分段，
    所以整张图跑完才重置一次。第一段采样把它们撑大后，第二段的上下文编码就
    拿不到显存放编码器权重，只能从系统内存反复搬运（实测 44 秒劣化到十几分钟，
    GPU 显示 100% 利用率却只有 30W）。
    """
    def _free_gb():
        try:
            return mm.get_free_memory() / (1024 ** 3)
        except Exception:
            return None

    def _aimdo_gb():
        try:
            import comfy_aimdo.control as _ctl
            return _ctl.get_total_vram_usage() / (1024 ** 3)
        except Exception:
            return None

    def _reclaimable_gb():
        """aimdo 自己认为「能还回来」的量（get_free_memory 会把它算进可用显存）。"""
        try:
            import comfy_aimdo.model_vbar as _mv
            return _mv.vbars_analyze() / (1024 ** 3)
        except Exception:
            return None

    def _castbuf_info():
        """(cast buffer 个数, 总字节)。VRAMBuffer 只增不减，靠 clear 触发 __del__。"""
        try:
            cbs = getattr(mm, "STREAM_AIMDO_CAST_BUFFERS", None) or {}
            return len(cbs), sum(int(b.size()) for b in cbs.values() if b is not None)
        except Exception:
            return -1, 0

    before = _free_gb()
    aimdo_before = _aimdo_gb()
    done = []
    fn = getattr(mm, "reset_cast_buffers", None)
    if fn is not None:
        try:
            fn()
            done.append("cast_buffers")
        except Exception as exc:
            log.debug("reset_cast_buffers skipped: %s", exc)

    try:
        import comfy_aimdo.model_vbar as _mvbar
        _mvbar.vbars_reset_watermark_limits()
        done.append("vbar_watermarks")
    except Exception as exc:
        log.debug("vbars_reset_watermark_limits skipped: %s", exc)

    # ---- 第三层：把「已卸载模型」的 VBAR 降优先级并显式放页
    #
    # 这是原先漏掉的一层。``comfy/model_patcher.py`` 在**每次**动态模型加载时
    # 调 ``vbar.prioritize()``，而 aimdo 文档写明：
    #   "Using prioritize resets the offload watermark of that model to
    #    **no offloading**, giving its weights priority over any other
    #    currently loaded models."
    # 全仓库**没有任何地方**调 ``deprioritize()``（已 grep 确认）。于是生成模型
    # 在第一段采样时拉高的优先级一直留着，它的驻留页被保护、不可驱逐；
    # 第二段编码器只能 fault 到 0 页 → 每层现读硬盘 → 34 秒变 8 分钟。
    #
    # 交换是划算的：生成模型下一段要重新 fault 自己的权重，但它走的是**带预取**
    # 的采样路径（``model_base.py`` 把 prefetch_dynamic_vbars 设为 True），
    # 低显存下本来就能正常跑；而编码器没有预取、对显存短缺零容忍。
    still_loaded = set()
    try:
        for lm in list(getattr(mm, "current_loaded_models", []) or []):
            still_loaded.add(id(getattr(lm, "model", None)))
    except Exception:
        pass

    n_dep = 0
    n_pages = 0
    n_unpin = 0
    n_ratchet = 0
    n_detached_mod = 0
    n_detached_vbar = 0
    freed = 0
    stuck = []          # 常规释放一点没放动的，留给最后强制卸除
    for patcher, vbar in vbars:
        if patcher is not None and id(patcher) in still_loaded:
            continue  # 还在场上的模型不动它
        # patcher 为 None 的是注册表兜底找到的孤儿 VBAR：它的模型早已不在场，
        # 但对象还活着、页还占着显存 —— 正是段 2 拿不到窗口的原因。
        try:
            vbar.deprioritize()
            n_dep += 1
        except Exception as exc:
            log.debug("vbar.deprioritize skipped: %s", exc)
        # 必须先解 pin：pin 着的页 free_memory 拿不走（模拟实测返回 0）
        _u, _r = _unpin_model(patcher)
        n_unpin += _u
        n_ratchet += _r
        try:
            got = int(vbar.free_memory(1 << 62))
        except Exception as exc:
            log.debug("vbar.free_memory skipped: %s", exc)
            got = 0
        if got > 0:
            freed += got
            n_pages += 1
        else:
            stuck.append((patcher, vbar))

    if n_dep:
        done.append("vbar_deprioritize x%d" % n_dep)
    if n_pages:
        done.append("vbar_pages %.2fG x%d" % (freed / (1024 ** 3), n_pages))

    after = _free_gb()
    aimdo_after = _aimdo_gb()

    # ★ [诊断] 三层释放全部落空：每个 VBAR 都报 loaded=0.00G 却 pages_freed=0.00G，
    #   说明那几 GB 压根不在 VBAR 里。不再猜，直接让 aimdo 自己把账目打出来。
    #   _castbuf_info / _reclaimable_gb 早就写好了却从没被调用过，这里接上。
    castbuf_n, castbuf_bytes = _castbuf_info()
    reclaim_gb = _reclaimable_gb()
    try:
        import torch as _torch
        true_free_gb = _torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        true_free_gb = None
    try:
        import comfy_aimdo.control as _ctl
        _ctl.analyze()          # 让 DLL 打印它认为的显存构成
    except Exception as exc:
        log.debug("aimdo analyze skipped: %s", exc)

    # 逐个 VBAR 报身份 —— 不报的话根本不知道 4.5 G 是谁攥着的
    for patcher, vbar in vbars:
        try:
            name = type(getattr(patcher, "model", patcher)).__name__
        except Exception:
            name = "?"
        try:
            loaded_gb = int(vbar.loaded_size()) / (1024 ** 3)
        except Exception:
            loaded_gb = float("nan")
        try:
            pages = int(vbar.get_nr_pages())
        except Exception:
            pages = -1
        try:
            mark = int(vbar.get_watermark())
        except Exception:
            mark = -1
        try:
            kept = patcher is not None and id(patcher) in still_loaded
        except Exception:
            kept = False
        log.info(
            "H3Director[vram]   vbar %s: loaded=%.2fG pages=%d watermark=%d kept=%s",
            name, loaded_gb, pages, mark, kept,
        )

    n_vbars_total = len(vbars)

    # ⚠⚠ 2026-09-19 实测：强制卸除这条路走不通，默认关闭。⚠⚠
    #
    # 开启后的一次完整复现：
    #   20:26:21  detached_vbars=3        → ModelVBAR.__del__ → vbar_free()
    #   20:26:22  重新加载 MiniMaxH3VideoVAE
    #   20:26:24  重新加载 MiniMaxH3TEModel_，文本编码器 forward 一开始
    #   → Fatal Python error: Aborted
    #     @ comfy/ops.py:796 forward_comfy_cast_weights
    #
    # 原因：本函数只确认了 patcher.model.modules() 上的 _v 已清空，但**别处仍持有
    # 旧 VBAR 地址空间的指针** —— prefetch 队列、CROSS_STEP_STATE、cast buffer 缓存、
    # CUDA graph 捕获……vbar_free() 一执行，那些就成了野指针。
    # 原先写的"leftover==0 就安全"这个闸门**不够**，它只覆盖了模块属性这一处。
    #
    # 唯一确定的正向结论（值得记下来）：
    #   detached_vbars=3 而 aimdo_vram_after_detach 仍是 4.02G
    #   → 那 4 GB **不在任何 VBAR 里**。摘 VBAR 这个方向可以彻底排除了。
    DETACH_VBAR_AFTER_RELEASE = False

    _p = _v = None
    if DETACH_VBAR_AFTER_RELEASE:
        for _p, _v in stuck:
            _c, _d, _ok = _detach_vbar(_p, _v)
            n_detached_mod += _c
            n_detached_vbar += _d
    _p = _v = None            # 放掉循环变量残留的引用

    after2 = aimdo_after2 = None
    if n_detached_vbar:
        # 关键：本函数自己还攥着两份引用（stuck 和 vbars 里的元组）。
        # 不清掉的话引用计数归不了零，__del__ 永远不跑，摘了也白摘。
        stuck[:] = []
        try:
            vbars[:] = []
        except Exception:
            pass
        gc.collect()
        after2 = _free_gb()
        aimdo_after2 = _aimdo_gb()

    log.info(
        "H3Director[vram] aimdo release: %s | vbars=%d still_loaded=%d "
        "deprioritized=%d unpinned=%d ratchet_cleared=%d pages_freed=%.2fG | "
        "vram_free %.2fG -> %.2fG | aimdo_vram %.2fG -> %.2fG | "
        "castbuf=%dx%.2fG reclaimable=%.2fG true_free_vram=%.2fG | "
        "detach_mod=%d detached_vbars=%d vram_free_after_detach=%.2fG "
        "aimdo_vram_after_detach=%.2fG",
        "+".join(done) if done else "none",
        n_vbars_total,
        len(still_loaded),
        n_dep,
        n_unpin,
        n_ratchet,
        freed / (1024 ** 3),
        before if before is not None else float("nan"),
        after if after is not None else float("nan"),
        aimdo_before if aimdo_before is not None else float("nan"),
        aimdo_after if aimdo_after is not None else float("nan"),
        castbuf_n,
        castbuf_bytes / (1024 ** 3),
        reclaim_gb if reclaim_gb is not None else float("nan"),
        true_free_gb if true_free_gb is not None else float("nan"),
        n_detached_mod,
        n_detached_vbar,
        after2 if after2 is not None else float("nan"),
        aimdo_after2 if aimdo_after2 is not None else float("nan"),
    )


def cleanup_segment_vram(
    *,
    enabled: bool = True,
    unload_models: bool = True,
    keep=(),
    adaptive: bool = True,
    pool=None,
) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache.

    ``keep`` 是**要保留在场**的模型（传 ModelPatcher 即可，内部会映射到
    LoadedModel）。典型用法是按阶段只保留该阶段用得上的模型，其余卸掉 ——
    这样相邻阶段所需的大模型不会同时挤在内存里。
    """
    if not enabled:
        return
    if unload_models and adaptive and not should_evict(pool):
        log.debug(
            "H3Director[vram] skip segment cleanup (loaded models fit in RAM)"
        )
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        _evict_dead_loaded_models()
        vbars = _collect_vbars(mm, pool)
        if unload_models:
            if keep:
                unload_except(keep)
            else:
                mm.unload_all_models()
            mm.cleanup_models()
        _evict_dead_loaded_models()
        gc.collect()

        # ★ 对齐 ComfyUI 的「每节点重置」。
        #
        # execution.py:546-554 的 finally 里一共做四件事：
        #     analyze()                      （仅 --verbose DEBUG 时）
        #     cleanup_prefetch_queues()      ★ 导演台原先从来没做过这一步
        #     reset_cast_buffers()           ← 后面 _release_aimdo_vram 里做了
        #     vbars_reset_watermark_limits() ← 同上
        #
        # 为什么漏掉的这一步是要害：预取队列持有模块引用和 fault 出来的
        # ``_v_block`` 地址范围，而导演台是「一次节点执行内循环所有分段」，
        # 这套重置整条片子只跑一次 —— 段1 采样撑起来的预取状态没人清，
        # 段2 起的编码器就一直拿不到显存窗口。
        # 小莫抓栈时命中的 ``prefetch_queue_pop``（llama.py:908）正是这条路径。
        #
        # 官方是在**每个节点之后**无条件调用它，所以放在段边界调用是安全的。
        try:
            import comfy.model_prefetch as _mp
            _mp.cleanup_prefetch_queues()
            log.info("H3Director[vram] prefetch queues cleaned")
        except Exception as exc:
            log.debug("cleanup_prefetch_queues skipped: %s", exc)

        mm.soft_empty_cache()
        _release_aimdo_vram(mm, vbars=vbars)
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        kept = len(tuple(keep or ()))
        log.info(
            "H3Director[vram] cleanup done: unloaded all except %d kept",
            kept,
        )
    else:
        log.info("H3Director[vram] cleanup done: cache cleared, models kept loaded")
