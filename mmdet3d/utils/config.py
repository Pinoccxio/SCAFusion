import copy

__all__ = ["recursive_eval"]


def recursive_eval(obj, globals=None):
    """递归解析配置对象中的表达式
    
    该函数用于递归解析配置对象(字典、列表等)中的表达式。
    对于以"${"开头,"}"结尾的字符串,会将其作为Python表达式进行求值。
    
    Args:
        obj: 需要解析的配置对象,可以是字典、列表或字符串等
        globals: 全局变量字典,用于表达式求值时的命名空间。默认为None,
                会使用obj的深拷贝作为全局变量
    
    Returns:
        解析后的配置对象
    """
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        # 递归解析字典中的每个值
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        # 递归解析列表中的每个元素
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        # 解析并执行"${...}"形式的表达式
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj
