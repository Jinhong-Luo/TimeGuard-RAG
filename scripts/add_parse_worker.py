"""停用在线扩容入口，避免 NVIDIA legacy hook 的设备访问丢失。"""

raise SystemExit(
    "已停用：运行中 GPU 容器执行 docker update 可能丢失设备访问权限。"
    "请先保存任务和消费者快照，再按审计计划受控重建固定资源的本项目服务；"
    "每个执行器必须通过真实 CUDA 预检后才允许领取任务。"
)
