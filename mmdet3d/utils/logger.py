import logging
from mmcv.utils import get_logger

#? train.py中使用
def get_root_logger(log_file=None, log_level=logging.INFO, name="mmdet3d"):
    """Get root logger and add a keyword filter to it.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    # 当log_file为空时，日志只会输出到控制台
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1  # 可以被顺利执行
    # 替代方案：logging_filter.filter = lambda record: name in record.name # 只保留包含name的记录

    return logger
