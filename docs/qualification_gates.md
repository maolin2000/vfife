# 单元与材料验证准入门槛

能力注册回答“软件实现声称支持什么”；qualification 回答“这些声明具有什么等级的证据”。两者不得混用。

## 单元门槛

`verified` 至少需要：

- `rigid_body`：刚体平移/转动不产生内力；
- `fundamental_response`：patch test、解析解或等价基本力学响应；
- `equilibrium`：节点力、反力或整体平衡；
- 杆系/混合维模型还需要 `mixed_assembly`。

`benchmarked` 进一步需要：

- `mesh_convergence`；
- `dynamic`；
- `restart`；
- `external_reference`。

`production` 还需要：

- 所有已声明后端的 `backend_parity`；
- `failure_path`，证明非法、失稳或非有限响应会受控失败。

## 材料门槛

`verified` 的基础门槛为：

- `monotonic`；
- `step_convergence`；
- `energy`。

条件门槛：

- 非线性材料必须有 `unload_reverse`；
- 三维材料必须有 `multiaxial`；
- 正则化材料必须有 `mesh_objectivity`。

`benchmarked` 增加 `restart` 和 `external_reference`；`production` 再增加全部声明后端的 `backend_parity` 与 `failure_path`。

## 证据原则

- 证据引用 pytest node id、verification contract 或公开 URL；
- 文件引用必须真实存在；
- 一个测试可以证明多个门槛，但每项含义必须确实被该测试断言；
- 公开文献只证明参考来源，不能替代本软件结果比较；
- GPU 仅“能够运行”不能算后端一致性证据；
- 缺少证据时状态是 `INCOMPLETE`，不得自动降级或沿用旧描述。

当前已建立 `hexa8`、`beam2`、`beam2_timoshenko`、`connector2`、E03 `shell4`、Chaboche、CDP 和 M02 `piecewise_uniaxial` 的 verified 清单。其余模型仍是“已注册、已有零散测试、尚未通过新清单整理”的状态，详见 [qualification/README.md](../qualification/README.md)。
