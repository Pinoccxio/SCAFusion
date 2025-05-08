import mmcv


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline. #? 从pipeline中返回的dict
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor | None: Data term.
    """
    # 如果key不在results字典中,返回None
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    # 如果数据是列表或元组类型,取第一个元素
    if isinstance(data, (list, tuple)):
        data = data[0]
    # 如果数据被DataContainer包装,提取其中的数据
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data
