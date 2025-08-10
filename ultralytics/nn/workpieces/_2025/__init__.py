import pkgutil
import importlib

__all__ = []

# 动态导入所有子模块的公开对象
package_dir = __path__
for (_, module_name, _) in pkgutil.iter_modules(package_dir):
    try:
        # 尝试导入模块
        module = importlib.import_module(f".{module_name}", __package__)
        
        # 遍历模块的 __all__ 属性中的所有对象
        for name in getattr(module, '__all__', []):
            try:
                # 获取对象并添加到全局命名空间
                obj = getattr(module, name)
                globals()[name] = obj
                __all__.append(name)
            except AttributeError:
                # 如果 __all__ 中定义的名称在模块中不存在，跳过该对象
                continue
                
    except (ImportError, ModuleNotFoundError, AttributeError, SyntaxError) as e:
        # 如果模块导入失败，打印警告并跳过该模块
        continue
    except Exception as e:
        # 捕获其他未预期的异常
        continue
