
# API参考

<cite>
**本文档中引用的文件**  
- [model/base.py](file://qlib/model/base.py)
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py)
- [strategy/base.py](file://qlib/strategy/base.py)
- [backtest/executor.py](file://qlib/backtest/executor.py)
- [workflow/recorder.py](file://qlib/workflow/recorder.py)
- [workflow/__init__.py](file://qlib/workflow/__init__.py)
- [model/trainer.py](file://qlib/model/trainer.py)
</cite>

## 目录
1. [简介](#简介)
2. [核心类概述](#核心类概述)
3. [Model类](#model类)
4. [Dataset类](#dataset类)
5. [Strategy类](#strategy类)
6. [Executor类](#executor类)
7. [Recorder类](#recorder类)
8. [使用示例](#使用示例)

## 简介
Qlib是一个用于量化投资研究的机器学习框架，提供了一套完整的API来支持从数据获取、模型训练到回测和策略执行的全流程。本参考文档详细介绍了Qlib的核心公共API，包括Model、Dataset、Strategy、Executor和Recorder等关键类的参数、返回值、异常和使用方法。

## 核心类概述
Qlib框架的核心组件包括：
- **Model**: 机器学习模型基类，定义了模型训练和预测的接口
- **Dataset**: 数据集基类，负责数据准备和处理
- **Strategy**: 交易策略基类，生成交易决策
- **Executor**: 执行器基类，执行交易决策并管理交易过程
- **Recorder**: 记录器类，用于实验管理和结果记录

这些组件通过清晰的接口设计实现了模块化和可扩展性。

## Model类

### Model
Model类是所有机器学习模型的基类，定义了模型训练和预测的基本接口。

**方法**

#### fit
训练模型。

**参数**
- dataset: Dataset - 用于训练的数据集
- reweighter: Reweighter - 重加权器，可选

**返回值**
- None

**异常**
- NotImplementedError - 子类必须实现此方法

**Section sources**
- [model/base.py](file://qlib/model/base.py#L24-L59)

#### predict
对给定数据集进行预测。

**参数**
- dataset: Dataset - 用于预测的数据集
- segment: Union[Text, slice] - 数据段，默认为"test"

**返回值**
- object - 预测结果，如pandas.Series

**异常**
- NotImplementedError - 子类必须实现此方法

**Section sources**
- [model/base.py](file://qlib/model/base.py#L62-L77)

### ModelFT
可微调模型基类，支持在已有模型基础上进行微调。

**方法**

#### finetune
基于给定数据集对模型进行微调。

**参数**
- dataset: Dataset - 用于微调的数据集

**返回值**
- None

**异常**
- NotImplementedError - 子类必须实现此方法

**Section sources**
- [model/base.py](file://qlib/model/base.py#L80-L109)

## Dataset类

### Dataset
数据集基类，负责数据准备和处理。

**方法**

#### __init__
初始化数据集。

**参数**
- **kwargs - 传递给setup_data的参数

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L14-L68)

#### config
配置数据集参数。

**参数**
- **kwargs - 配置参数

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L34-L38)

#### setup_data
设置数据。

**参数**
- **kwargs - 设置参数

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L40-L53)

#### prepare
准备用于模型训练或推理的数据。

**参数**
- segments: Union[List[Text], Tuple[Text], Text, slice, pd.Index] - 数据段描述
- col_set: str - 列集合，默认为DataHandler.CS_ALL
- data_key: str - 数据键，默认为DataHandlerLP.DK_I
- **kwargs - 其他参数

**返回值**
- Union[List[pd.DataFrame], pd.DataFrame] - 准备好的数据

**异常**
- NotImplementedError - 子类必须实现此方法

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L55-L68)

### DatasetH
带有DataHandler的数据集类。

**方法**

#### __init__
初始化DatasetH实例。

**参数**
- handler: Union[Dict, DataHandler] - 数据处理器
- segments: Dict[Text, Tuple] - 数据段描述
- fetch_kwargs: Dict - 获取数据的参数
- **kwargs - 其他参数

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L71-L268)

#### config
初始化DatasetH。

**参数**
- handler_kwargs: dict - DataHandler的配置
- **kwargs - DatasetH的配置

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L71-L268)

#### setup_data
设置数据。

**参数**
- handler_kwargs: dict - DataHandler的初始化参数
- **kwargs - 其他参数

**返回值**
- None

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L71-L268)

#### prepare
准备数据。

**参数**
- segments: Union[List[Text], Tuple[Text], Text, slice, pd.Index] - 数据段描述
- col_set: str - 列集合
- data_key: str - 数据键
- **kwargs - 其他参数

**返回值**
- Union[List[pd.DataFrame], pd.DataFrame] - 准备好的数据

**Section sources**
- [data/dataset/__init__.py](file://qlib/data/dataset/__init__.py#L71-L268)

## Strategy类

### BaseStrategy
交易策略基类。

**方法**

#### __init__
初始化策略。

**参数**
- outer_trade_decision: BaseTradeDecision - 外部交易决策，可选
- level_infra: LevelInfrastructure - 层级基础设施，可选
- common_infra: CommonInfrastructure - 共享基础设施，可选
- trade_exchange: Exchange - 交易交易所

**返回值**
- None

**Section sources**
- [strategy/base.py](file://qlib/strategy/base.py#L22-L236)

#### generate_trade_decision
在每个交易周期生成交易决策。

**参数**
- execute_result: list - 执行结果，可选

**返回值**
- Union[BaseTradeDecision, Generator[Any, Any, BaseTradeDecision]] - 交易决策

**异常**
- NotImplementedError - 子类必须实现此方法

**Section sources**
- [strategy/base.py](file://qlib/strategy/base.py#L22-L236)

#### reset
重置策略。

**参数**
- level_infra: LevelInfrastructure - 层级基础设施，可选
- common_infra: CommonInfrastructure - 共享基础设施，可选
- outer_trade_decision: BaseTradeDecision - 外部交易决策，可选
- **kwargs - 其他参数

**返回值**
- None

**Section sources**
- [strategy/base.py](file://qlib/strategy/base.py#L22-L236)

## Executor类

### BaseExecutor
基础执行器类。

**方法**

#### __init__
初始化执行器。

**参数**
- time_per_step: str - 每步交易时间
- start_time: Union[str, pd.Timestamp] - 开始时间，可选
- end_time: Union[str, pd.Timestamp] - 结束时间，可选
- indicator_config: dict - 指标配置
- generate_portfolio_metrics: bool - 是否生成投资组合指标
- verbose: bool - 是否打印交易信息
- track_data: bool - 是否生成交易决策
- trade_exchange: Exchange | None - 交易交易所，可选
- common_infra: CommonInfrastructure | None - 共享基础设施，可选
- settle_type: str - 结算类型
- **kwargs: Any - 其他参数

**返回值**
- None

**Section sources**
- [backtest/executor.py](file://qlib/backtest/executor.py#L21-L306)

#### execute
执行交易决策。

**参数**
- trade_decision: BaseTradeDecision - 交易决策
- level: int - 当前执行器层级，默认为0

**返回值**
- List[object] - 执行结果

**Section sources**
- [back