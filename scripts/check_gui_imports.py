import importlib
modules = ['customtkinter','matplotlib','PIL','trimesh','numpy','xatlas']
miss = []
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        miss.append((m, str(e)))
print('MISSING:', miss)
