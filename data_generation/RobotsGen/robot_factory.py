"""
    most of this code is taken from tutorial "create_some_modules.ipynb"
    in ident/timor-python/tutorials/create_some_modules.ipynb
    and the reference notebook "planar_factory_2d.py
"""
import numpy as np
import os

from timor.Bodies import Body, Connector, Gender
from timor.Joints import Joint
from timor.Module import AtomicModule, ModulesDB, ModuleHeader
from timor.Geometry import Box, ComposedGeometry, Cylinder, Sphere
from timor.utilities.transformation import Transformation
from timor.utilities.spatial import rotX, rotY, rotZ

# Let's define some principle parameters
diameter = 80 / 1000  # the diameter of our links [m]
inner_diameter = 60 / 1000  # assume the links are hollow, with 5cm thick walls
connector_size = .11  # Doesn't matter as long as it is the same for all connectors
# sizes = (150 / 1000, 300 / 1000, 450 / 1000)  # we are going to build links in various lengths (sizes) [m]

ROT_X = Transformation.from_rotation(rotX(np.pi)[:3, :3])  # A shortcut we are going to use frequently
ROT_Y = Transformation.from_rotation(rotY(np.pi)[:3, :3])  # A shortcut we are going to use frequently


def base() -> ModulesDB:
    """
    Creates a base connector attached to an empty body.
    The output connector is pointing in positive x-direction to keep it basically planar.
    """
    length = 0.01  # Geometry can NOT be empty, otherwise a value error occurs when creating/importing a json file
    geometry = Box({'x': length, 'y': length, 'z': length})  # pose=Transformation.from_translation([0, 0, l / 2])
    c_world = Connector('base', ROT_X @ Transformation.from_translation([0, 0, length / 2]), gender=Gender.f,
                        connector_type='base', size=connector_size)
    # How to orient the connector?
    # - It has to "point away" from our module
    # - The z-axis of the connector should be aligned with the global x-axis
    # - We assume the body coordinate system to be aligned with the global coordinate system for the base, so:
    c_robot = Connector('base2robot', gender=Gender.m, connector_type='default', size=connector_size,
                        body2connector=Transformation.from_translation(
                            [length / 2, 0, -length / 2]) @ rotY(np.pi / 2) @ rotZ(np.pi))
    # The rotZ is just "cosmetics"
    return ModulesDB({AtomicModule(ModuleHeader(ID='base', name='Base'), [Body('base', collision=geometry,
                                                                               connectors=[c_world, c_robot])])})


def eef() -> ModulesDB:
    """Creates an end effector that's basically just a small sphere (nice for visualization)."""
    diameter = .1
    geometry = Sphere({'r': diameter}, pose=Transformation.from_translation([0, 0, diameter / 2]))
    c_robot = Connector('robot2eef', ROT_X, gender=Gender.f, connector_type='default', size=connector_size)
    c_world = Connector('end-effector', gender=Gender.m, connector_type='eef',
                        body2connector=Transformation.from_translation([0, 0, diameter / 2]))
    return ModulesDB({AtomicModule(ModuleHeader(ID='eef', name='Demo EEF'), [Body('EEF',
                                                                                  collision=geometry,
                                                                                  connectors=[c_robot, c_world])])})


def i_links(sizes) -> ModulesDB:
    """Create I links with a certain lengths."""
    links = ModulesDB()
    diameter = .1

    for size in sizes:
        module_header = ModuleHeader(ID='i_{}'.format(int(size * 100)), name='I shaped link {}'.format(int(size * 100)))
        connectors = (
            Connector(f'{int(diameter * 100)}-{i}',
                      ROT_X @ Transformation.from_translation([0, 0, size / 2]) if i == 0
                      else Transformation.from_translation([0, 0, size / 2]),
                      gender=Gender.f if i == 0 else Gender.m,
                      connector_type='default',
                      size=connector_size)
            for i in range(2)
        )
        geometry = Cylinder({'r': diameter, 'z': size}, pose=Transformation.from_translation([0, 0, 0]))
        body = Body('i_{}'.format(int(size * 100)), collision=geometry, connectors=connectors)
        links.add(AtomicModule(module_header, [body]))
    return links


def l_links_xy_plane(sizes) -> ModulesDB:
    """For every size, creates an I-shaped link (aka two orthogonal cylinders) with two connectors."""
    links = ModulesDB()
    offset = 100 / 1000
    diameter = .1
    for size in sizes:
        module_header = ModuleHeader(ID='l_{}'.format(int(size * 100)),
                                     name='L shaped link {}-{}-{}'.format(offset, diameter, int(size * 100)))

        body2con1 = ROT_Y
        body2con2 = Transformation.from_translation(
            [0, 0, offset]) @ Transformation.from_rotation(
            rotX(1.5 * np.pi)[:3, :3]) @ Transformation.from_translation([0, 0, size])
        # The distal connector is placed at the end of the second cylinder...

        connectors = (
            Connector(f'{int(diameter * 100)}-{i}',
                      body2con1 if i == 0 else body2con2,
                      gender=Gender.f if i == 0 else Gender.m,
                      connector_type='default',
                      size=connector_size)
            for i in range(2)
        )

        cyl1 = Cylinder({'r': diameter, 'z': offset},
                        pose=Transformation.from_translation([0, 0, offset / 2]))
        cyl2 = Cylinder({'r': diameter, 'z': size},
                        pose=body2con2 @ Transformation.from_translation([0, 0, - size / 2]))
        elbow = Sphere({'r': diameter}, pose=Transformation.from_translation([0, 0, offset]))
        # This "elbow" connects the cylinders - as we ignore the inertia, we can define it as a full sphere
        geometry = ComposedGeometry((cyl1, cyl2, elbow))  # We combine all bodies to one common geometry

        body = Body('l_{}'.format(int(100 * size)), collision=geometry, connectors=connectors)
        links.add(AtomicModule(module_header, [body]))
    return links


def joint() -> ModulesDB:
    # A revolutional joint is consist of two i linkes with lenght 1
    length = 0
    diameter = .1
    proximal = Body('J1_proximal', collision=Cylinder({'r': diameter, 'z': length}),
                    connectors=[Connector('J1_proximal', ROT_X, gender=Gender.f, connector_type='default',
                                          size=connector_size)]
                    )
    distal = Body('J1_distal', collision=Cylinder({'r': diameter, 'z': length}),
                  connectors=[Connector('J1_distal', Transformation.neutral(), gender=Gender.m,
                                        connector_type='default', size=connector_size)],
                  )
    joint = Joint(
        joint_id='joint',
        joint_type='revolute',
        parent_body=proximal,
        child_body=distal,
        q_limits=[-np.pi, np.pi],  # This is arbitrary
        parent2joint=rotY(-np.pi / 2),  # Make the rotation axis the global z-Axis
        joint2child=rotY(np.pi / 2)  # Rotate back to the plane we are "living" in
    )

    module_header = ModuleHeader(ID='J1', name='Revolute Joint')
    return ModulesDB({
        AtomicModule(module_header, [proximal, distal], [joint])
    })


def visible_joint_z() -> ModulesDB:
    # A revolutional joint is consist of two i linkes with lenght 1
    length = .1
    diameter = .1
    proximal = Body('J1_proximal', collision=Cylinder({'r': diameter, 'z': length}),
                    connectors=[Connector('J1_proximal',
                                          ROT_X @ Transformation.from_translation([0, 0, length / 2]),
                                          gender=Gender.f, connector_type='default', size=connector_size)]
                    )
    distal = Body('J1_distal', collision=Cylinder({'r': diameter, 'z': length}),
                  connectors=[Connector('J1_distal', Transformation.from_translation([0, 0, length / 2]),
                                        gender=Gender.m, connector_type='default', size=connector_size)],
                  )
    joint = Joint(
        joint_id='joint',
        joint_type='revolute',
        parent_body=proximal,
        child_body=distal,
        q_limits=[-np.pi, np.pi],  # This is arbitrary
        parent2joint=Transformation.from_translation(
            [0, 0, length / 2]) @ Transformation.from_rotation(rotY(-np.pi / 2)[:3, :3]),
        # Make the rotation axis the global z-Axis
        joint2child=Transformation.from_rotation(
            rotY(np.pi / 2)[:3, :3]) @ Transformation.from_translation([0, 0, length / 2])
        # Rotate back to the plane we are "living" in
    )

    module_header = ModuleHeader(ID='J1', name='Revolute Joint')
    return ModulesDB({
        AtomicModule(module_header, [proximal, distal], [joint])
    })


def visible_joint_y() -> ModulesDB:
    # A revolutional joint is consist of two i linkes with lenght 1
    length = .1
    diameter = .1
    proximal = Body('J2_proximal', collision=Cylinder({'r': diameter, 'z': length}),
                    connectors=[Connector('J1_proximal',
                                          ROT_X @ Transformation.from_translation([0, 0, length / 2]), gender=Gender.f,
                                          connector_type='default', size=connector_size)]
                    )
    distal = Body('J2_distal', collision=Cylinder({'r': diameter, 'z': length}),
                  connectors=[Connector('J1_distal', Transformation.from_translation([0, 0, length / 2]),
                                        gender=Gender.m, connector_type='default', size=connector_size)],
                  )
    joint = Joint(
        joint_id='joint',
        joint_type='revolute',
        parent_body=proximal,
        child_body=distal,
        q_limits=[-np.pi, np.pi],  # This is arbitrary
        parent2joint=Transformation.from_translation(
            [0, 0, length / 2]) @ Transformation.from_rotation(rotX(-np.pi / 2)[:3, :3]),
        # Make the rotation axis the global z-Axis
        joint2child=Transformation.from_rotation(
            rotX(np.pi / 2)[:3, :3]) @ Transformation.from_translation([0, 0, length / 2])
        # Rotate back to the plane we are "living" in
    )

    module_header = ModuleHeader(ID='J2', name='Revolute Joint Y')
    return ModulesDB({
        AtomicModule(module_header, [proximal, distal], [joint])
    })


def create_modules_for_2d(sizes, save=False, root_path=None) -> ModulesDB:
    db = base().union(eef()).union(i_links(sizes)).union(l_links_xy_plane(sizes)).union(visible_joint_z())

    if save:
        file_name = 'Modules_2D'
        for size in sizes:
            file_name += f'_{size * 100}'
        db.to_json_file(os.path.join(root_path, file_name))

    return db


def create_modules_for_3d(sizes, save=False, root_path=None) -> ModulesDB:
    db = base().union(eef()).union(i_links(sizes)).union(visible_joint_z()).union(visible_joint_y())

    if save:
        file_name = 'Modules_3D'
        for size in sizes:
            file_name += f'_{size * 100}'
        db.to_json_file(os.path.join(root_path, file_name))

    return db


if __name__ == '__main__':
    # root = os.path.dirname(os.path.abspath(__file__))
    # create_modules_for_2d([.2,.5], save=True, root_path=root)
    # file_path = os.path.join(root, "Modules_2D_20.0_50.0")
    # #with open(file_path, "r") as f:
    # #        db_import = ModulesDB.from_json_data(json.load(f), package_dir=None)
    # db_import = ModulesDB.from_file(filepath=file_path, package_dir=None)
    db = create_modules_for_2d([.2, .3])
    db.debug_visualization()
