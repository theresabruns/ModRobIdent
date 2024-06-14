import numpy as np
from numpy import ndarray
from timor import ModuleAssembly, ModulesDB
from timor.utilities.file_locations import get_module_db_files


def assembly_from_np_file(fn: str) -> ModuleAssembly:
    raw_data = np.load(fn)
    assembly = ModuleAssembly.from_json_data(raw_data)
    return assembly


def assembly_to_binary_encoding(assembly: ModuleAssembly, db: ModulesDB) -> ndarray:
    """
        create binary encoding of robot assembly
        underlying binary matrix:
        max_len = number of rows (maximum length of robot incl. base & eef)
        columns are all sorted modules of db (sorting by group & alphabetically within group:
            bases, links, joints, eefs
        )

        assembly: robot assembly to create the binary encoding for
        db: ModuleDB of which the assembly is based on
        max_len: maximum number of allowed modules in a robot
    """
    num_modules = len(db.all_module_ids)
    db_modules = db.all_module_ids

    sorted_db_module_ids = []  # bases + links + joints + eefs

    bases = db.bases.all_module_ids
    joints = set(map(lambda j: j.id[0], db.all_joints))
    eefs = db.end_effectors.all_module_ids
    links = db_modules.difference(bases).difference(joints).difference(eefs)

    sorted_db_module_ids += sorted(bases)
    sorted_db_module_ids += sorted(links)
    sorted_db_module_ids += sorted(joints)
    sorted_db_module_ids += sorted(eefs)

    orig = assembly.original_module_ids
    module_map = np.zeros((len(orig), num_modules))

    for i, mod_id in enumerate(orig):
        module_map[i][sorted_db_module_ids.index(mod_id)] = 1
    return module_map


if __name__ == '__main__':
    atomic_modules = ModulesDB.from_file(*get_module_db_files('geometric_primitive_modules'))
    simple_joint_assembly = ModuleAssembly.from_serial_modules(atomic_modules,
                                                               ['base', 'J2', 'l_15', 'i_15', 'J2', 'l_15', 'eef'])
    bn_encoding = assembly_to_binary_encoding(assembly=simple_joint_assembly, db=atomic_modules)
    print(bn_encoding)
    print(bn_encoding.shape)
