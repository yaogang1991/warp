# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 开发命令

### 构建和运行
- `cargo run` - 本地构建并运行 Warp
- `cargo bundle --bin warp` - 打包主应用
- `./script/run` - 通过便捷脚本构建并运行

### 本地服务器开发
连接到本地 warp-server 实例：
```bash
cargo run --features with_local_server
SERVER_ROOT_URL=http://localhost:8080 WS_SERVER_URL=ws://localhost:8080/graphql/v2 cargo run --features with_local_server
```

### 测试
- `cargo nextest run --no-fail-fast --workspace --exclude command-signatures-v2` - 使用 nextest 运行测试（并行执行）
- `cargo nextest run -p warp_completer --features v2` - 使用 v2 特性运行 completer 测试
- `cargo test --doc` - 运行文档测试
- `cargo test -p <crate>` - 运行指定包的测试

### 代码检查和格式化
- `./script/presubmit` - 运行所有提交前检查（fmt、clippy、tests）
- `cargo fmt` - 格式化代码
- `cargo clippy --workspace --all-targets --all-features --tests -- -D warnings` - 运行 clippy
- `./script/run-clang-format.py -r --extensions 'c,h,cpp,m' ./crates/warpui/src/ ./app/src/` - 格式化 C/C++/Obj-C 代码
- `find . -name "*.wgsl" -exec wgslfmt --check {} +` - 检查 WGSL 着色器格式

### 平台设置
- `./script/bootstrap` - 平台特定设置（调用平台特定的 bootstrap 脚本）

## 架构概述

Warp 是一个基于 Rust 的终端模拟器，使用名为 **WarpUI** 的自定义 UI 框架。

### 工作区结构
这是一个包含 60+ 成员 crate 的 Cargo 工作区。主要目录：
- `app/` - 主应用二进制文件；包含终端模拟、AI 集成、设置、工作区管理
- `crates/warpui/` 和 `crates/warpui_core/` - 自定义 UI 框架（实体-组件-句柄模式）
- `crates/warp_core/` - 核心工具和平台抽象
- `crates/warp_terminal/` - 终端模拟核心（PTY、网格渲染、块）
- `crates/editor/` - 文本编辑功能
- `crates/integration/` - 使用自定义框架的集成测试
- `crates/persistence/` - Diesel ORM 与 SQLite 本地存储
- `crates/graphql/` - 用于服务器通信的 GraphQL 客户端

### WarpUI 框架模式

**实体-句柄系统**：全局 `App` 对象拥有所有视图/模型（实体）。视图持有 `ViewHandle<T>` 引用其他视图，而非直接所有权。句柄通过 `AppContext` 在渲染/事件期间转换为引用。

```rust
struct WorkspaceView {
    sessions: Vec<ViewHandle<TerminalView>>,
}

impl View for WorkspaceView {
    fn render<'a>(&self, ..., ctx: &AppContext) -> ... {
        // 使用 context 将句柄转换为引用
        let title = self.sessions[0].as_ref(ctx).title();
    }
}
```

**上下文参数**：接受 `AppContext`、`ViewContext` 或 `ModelContext` 的函数应将参数命名为 `ctx` 并放在最后（闭包之前除外）。

**MouseStateHandle**：必须在构造期间创建一次并在其他地方克隆/引用。在渲染过程中内联使用 `MouseStateHandle::default()` 会破坏鼠标交互。

### 关键：终端模型锁定

**调用 `TerminalModel` 的 `model.lock()` 时要格外小心。** 从不同调用点获取同一模型的多个锁会导致死锁（UI 冻结）。

规则：
- 添加 `model.lock()` 前，验证当前调用栈中没有调用者已持有该锁
- 优先将已锁定的模型引用沿调用栈向下传递
- 保持锁定范围尽可能短
- 避免在锁内调用可能加锁的其他函数

### 功能标志

功能标志使用编译时定义和运行时检查：

1. 在 `crates/warp_features/src/lib.rs` 的 `FeatureFlag` 枚举中添加变体
2. 可选择在 `DOGFOOD_FLAGS` 中为 dogfood 构建启用
3. 使用 `FeatureFlag::YourFlag.is_enabled()` 控制代码
4. 优先使用运行时检查而非 `#[cfg(...)]`，以便更容易切换和清理

```rust
if FeatureFlag::YourNewFeature.is_enabled() {
    // 受控的行为
}
```

### 代码风格偏好
- 避免不必要的类型注解，尤其是在闭包参数中
- 优先使用导入而非完全限定的路径；将导入放在文件顶部
- 完全删除未使用的参数，而非用 `_` 前缀
- 使用内联格式参数：`println!("{x}")` 而非 `println!("{}", x)`
- 在 match 语句中尽可能避免 `_` 通配符 - 穷尽匹配有助于捕获遗漏的变体

### 测试约定
- 单元测试放在单独文件中：`${filename}_tests.rs` 或 `${mod}_test.rs`
- 在模块末尾包含测试文件：
  ```rust
  #[cfg(test)]
  #[path = "filename_tests.rs"]
  mod tests;
  ```
- 集成测试使用 `crates/integration/` 中的自定义 Builder/TestStep 框架

### 数据库
- 使用 Diesel ORM 和 SQLite
- 迁移文件在 `crates/persistence/migrations/`
- Schema 在 `crates/persistence/src/schema.rs`

## /sync-upstream — 同步上游并保持个人功能

此命令将官方 warpdotdev/warp 的最新代码同步到 `personal` 分支，确保本地 AI 等个人功能始终基于最新上游代码。

运行 `同步上游` 或 `/sync-upstream` 时执行以下步骤：

1. `git fetch upstream`（若 `upstream` remote 不存在则先 `git remote add upstream https://github.com/warpdotdev/warp.git`）
2. `git checkout personal`
3. `git merge upstream/master`
4. 若有冲突，逐个文件解决（优先保留 personal 分支的改动）
5. `git push yaogang personal`

前置条件：
- `upstream` remote → `https://github.com/warpdotdev/warp.git`
- `yaogang` remote → `https://github.com/yaogang1991/warp.git`
- `personal` 分支包含本地 AI 支持等个人功能

## 提交流程

完成代码修改后，按以下步骤提交并推送到个人 fork：

1. 在 `master` 分支上 `git add` 相关文件并 `git commit`
2. `git checkout personal && git merge master` — 将改动合并到 personal 分支
3. `git push yaogang personal` — 推送到 yaogang1991/warp 的 personal 分支
4. `git checkout master` — 切回 master 继续开发

前置条件（同 /sync-upstream）：
- `yaogang` remote → `https://github.com/yaogang1991/warp.git`
- `personal` 分支跟踪 yaogang 远程

### 按功能分类的关键模块
- `app/src/terminal/` - 终端视图、PTY 处理、输入、历史、块
- `app/src/workspace/` - 工作区容器、标签页、窗格、会话
- `app/src/ai/` - Agent 模式、AI 对话、技能、MCP 集成
- `app/src/settings/` - 用户设置和偏好
- `app/src/notebooks/` - 笔记本功能
- `crates/warp_completer/` - 命令补全引擎
