#!/usr/bin/env python3
import numpy as np
import operator

from enum import IntEnum


class VoxelAxes(IntEnum):
    labx = x = 1
    laby = y = 2
    labz = z = 3
    pcpx = -labx
    pcpy = -laby
    pcpz = -labz


class VoxelSpace:
    def __init__(self, extent: int) -> None:
        """
        Arguments:
            extent: The extent of the cubic latent space.
        Variables:
            self.extent: ...
            self.latent: The latent space.
        """
        self.extent = extent
        self.latent = np.indices((self.extent,) * 3)

    def shift(self, filled: np.ndarray, ijk: tuple) -> np.ndarray:
        """
        Shift a 3d array.

        Arguments:
            filled: A boolean array of filled locations.
            ijk: Translation to apply, determined by (0, 0, 0) |-> (i, j, k).
                Crops the filled array if it hits edges. Accept i, j, k > 0.
        Returns:
            The translated and cropped array.
        """
        if not all(q >= 0 for q in ijk):
            raise ValueError(f'Shift must be positive. Got: {ijk}')
        i, j, k = ijk
        rolled = np.roll(filled, shift=(i, j, k), axis=(0, 1, 2))
        cropped = rolled
        cropped[:i, :, :] = 0
        cropped[:, :j, :] = 0
        cropped[:, :, :k] = 0
        return cropped

    def rotate(self, filled: np.ndarray, ijk: tuple, ypr: tuple) -> np.ndarray:
        """
        Rotate a 3d array. WILL leave holes.

        Arguments:
            filled: A boolean array of filled locations.
            ijk: Point to rotate about.
            ypr: Rotation angles in about lab-frame (z, y, x).
        Returns:
            The rotated and cropped array (same shape as input)
        """
        y, p, r = ypr
        i, j, k = ijk

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(r), -np.sin(r)],
                [0, np.sin(r), np.cos(r)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(p), 0, np.sin(p)],
                [0, 1, 0],
                [-np.sin(p), 0, np.cos(p)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(y), -np.sin(y), 0],
                [np.sin(y), np.cos(y), 0],
                [0, 0, 1],
            ]
        )
        # Order: yaw <- pitch <- roll <- obj
        R = Rz @ Ry @ Rx

        rotated = np.zeros_like(filled)
        xyzs = np.argwhere(filled)

        for x, y, z in xyzs:
            x_c = x - i
            y_c = y - j
            z_c = z - k

            rotated_xyz = R @ np.array([x_c, y_c, z_c])
            x_rot = int(np.round(rotated_xyz[0] + i))
            y_rot = int(np.round(rotated_xyz[1] + j))
            z_rot = int(np.round(rotated_xyz[2] + k))

            if (
                0 <= x_rot < self.extent
                and 0 <= y_rot < self.extent
                and 0 <= z_rot < self.extent
            ):
                rotated[x_rot, y_rot, z_rot] = True

        return rotated

    @staticmethod
    def center_of_mass(vox: 'Voxel') -> tuple:
        """
        Calculate the center of mass.
        If mass is 0, the center of mass will be (0, 0, 0).

        Arguments:
            vox: The voxel object.
        Returns:
            The center of mass in lab frame.
        """
        densities = vox.densities
        M = vox.mass or 1  # M = 0 -> center of mass is (0, 0, 0)
        # Add 1/2 to each direction as we are actually calculating the center
        # of mass from the (0, 0, 0) corner.
        return tuple(
            ((qs * densities).sum()) / M + 1 / 2
            for qs in np.indices(densities.shape)
        )

    @staticmethod
    def inertia_tensor_c(vox: 'Voxel') -> np.ndarray:
        """
        Calculate the inertia tensor about the center of mass.

        Arguments:
            vox: The voxel object.
        Returns:
            3x3 lab frame inertia tensor about the center of mass.
        """
        densities = vox.densities
        x_c, y_c, z_c = vox.center_of_mass
        X, Y, Z = np.indices(densities.shape)
        x = X - x_c + 1 / 2
        y = Y - y_c + 1 / 2
        z = Z - z_c + 1 / 2
        Ixx = (densities * (y**2 + z**2)).sum()
        Iyy = (densities * (z**2 + x**2)).sum()
        Izz = (densities * (x**2 + y**2)).sum()
        Ixy = Iyx = -(densities * x * y).sum()
        Iyz = Izy = -(densities * y * z).sum()
        Izx = Ixz = -(densities * z * x).sum()
        return np.array(((Ixx, Ixy, Ixz), (Iyx, Iyy, Iyz), (Izx, Izy, Izz)))

    @staticmethod
    def principal_axes(vox: 'Voxel') -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate diagonalized inertia tensor and the Euler angles of principal
        axes.

        Arguments:
            vox: The voxel object.
        Returns:
            3x3 diagonal inertia tensor in principal frame, and the associated
            initial rotation matrix.
        """
        I_c = vox.inertia_tensor
        evals, evecs = np.linalg.eigh(I_c)

        # Return sorted
        _s = np.argsort(evals)
        return np.diag(evals[_s]), evecs[:, _s]

    @staticmethod
    def parallel_axis(
        vox: 'Voxel', about: tuple, I_c: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Calculate diagonalized inertia tensor and the Euler angles of principal
        axes.

        Arguments:
            vox: The voxel object.
        Returns:
            3x3 diagonal inertia tensor in principal frame, and the associated
            Euler angles.
        """
        if I_c is None:
            I_c = vox.inertia_tensor

        com = vox.center_of_mass
        r = np.array(about) - np.array(com)

        return I_c + vox.mass * (np.eye(3) * np.dot(r, r) - np.outer(r, r))


class Voxel:
    def __init__(self, vs: VoxelSpace) -> None:
        """
        Arguments:
            vs: The voxel space to operate in.
        Variables:
            self.vs: ...
            self.components: A list of boolean arrays of filled locations.
            self.component_rhos: A list of float arrays of location densities.
            self.component_cols: A list of float arrays of location colors.
        """
        self.default_empty = []
        # self.default_color = 'g'

        self.vs = vs
        self.components = self.default_empty.copy()
        self.component_rhos = self.default_empty.copy()
        self.component_cols = self.default_empty.copy()

    @property
    def filled(self) -> np.ndarray:
        """
        Returns:
            Array of filled locations.
        """
        return np.logical_or.reduce(self.components)

    @property
    def densities(self) -> np.ndarray:
        """
        Overlapping densities are summed.

        Returns:
            Array of densities of filled locations.
        """
        return np.sum(self.component_rhos, axis=0)

    @property
    def facecolors(self) -> np.ndarray:
        """
        For overlapping colors, the first is chosen.

        Returns:
            Array of colors.
        """
        stacked = np.stack(self.component_cols)
        mask = operator.ne(stacked, None)
        _fsts = np.argmax(mask, axis=0)
        p, q, r = np.indices(_fsts.shape)
        return stacked[_fsts, p, q, r]

    @property
    def mass(self) -> float:
        """
        Calculated from `self.densities'.

        Returns:
            Total mass.
        """
        return self.densities.sum()

    def _set_components(self, filled_list: list[np.ndarray]) -> None:
        """
        Intended for internal use only.

        Arguments:
            filled_list: A list of boolean arrays of filled locations.
        Side effects:
            Overwrite `self.components'.
        """
        self.components = filled_list

    def _set_component_rhos(self, densities_list: list[np.ndarray]) -> None:
        """
        Intended for internal use only.

        Arguments:
            densities_list: A list of float arrays of densities of filled
                locations.
        Side effects:
            Overwrite `self.component_rhos'.
        """
        self.component_rhos = densities_list

    def _set_component_cols(self, facecolors_list: list[np.ndarray]) -> None:
        """
        Intended for internal use only.

        Arguments:
            facecolors_list: A list of float arrays of colors of filled
                locations.
        Side effects:
            Overwrite `self.component_cols'.
        """
        self.component_cols = facecolors_list

    @property
    def center_of_mass(self) -> tuple:
        """
        See `VoxelSpace.center_of_mass'

        Returns:
            self.vs.center_of_mass(self)
        """
        return self.vs.center_of_mass(self)

    @property
    def inertia_tensor(self) -> np.ndarray:
        """
        See `VoxelSpace.inertia_tensor_c'

        Returns:
            self.vs.inertia_tensor_c(self)
        """
        return self.vs.inertia_tensor_c(self)

    def inertia_tensor_about(self, i: float, j: float, k: float) -> np.ndarray:
        """
        See `VoxelSpace.inertia_tensor_c', `VoxelSpace.parallel_axis'

        Returns:
            Inertia tensor about `about'.
        """
        I_c = self.vs.inertia_tensor_c(self)
        return self.vs.parallel_axis(self, (i, j, k), I_c)

    @property
    def principal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """
        See `VoxelSpace.principal_axes'

        Returns:
            self.vs.principal_axes(self)
        """
        return self.vs.principal_axes(self)

    def lay(self, locs: np.ndarray, rho: float, color: str) -> None:
        """
        Adds a component to the internal representation. Each individual
        component is of uniform density and color.

        Arguments:
            locs: A boolean array of filled locations to be added.
            rho: Density of the structure, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        self.components += [locs]
        self.component_rhos += [np.where(locs, rho, 0)]
        # `np.where' cannot set the dtype.
        cols = np.empty(locs.shape, dtype=object)
        cols[locs] = color
        self.component_cols += [cols]

    def make_point(
        self,
        ijk: tuple,
        ypr: tuple = (0, 0, 0),
        rho: float = 1.0,
        color: str = 'g',
    ):
        """
        Create a point.

        Arguments:
            ijk: Corner coordinates, determined by (0, 0, 0) |-> (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
                Does nothing.
            rho: Density of the prism, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        self.make_rect_prism((1, 1, 1), ijk, ypr, rho, color)

    def make_rect_prism(
        self,
        xyz: tuple,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a rectangular prism.

        Arguments:
            xyz: Extent of the prism.
            ijk: Corner coordinates, determined by (0, 0, 0) |-> (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
            rho: Density of the prism, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        X, Y, Z = self.vs.latent
        x, y, z = xyz
        i, j, k = ijk
        locs = self.vs.shift((X < x) & (Y < y) & (Z < z), ijk)

        ijk_c = (
            i + (x - 1) / 2,
            j + (y - 1) / 2,
            k + (z - 1) / 2,
        )

        if any(ypr):
            locs = self.vs.rotate(locs, ijk_c, ypr)

        self.lay(locs, rho, color)

    def make_sphere(
        self,
        r: float,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a sphere.

        Arguments:
            r: Radius of the sphere.
            ijk: Center coordinates (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
                Probably useless for a sphere.
            rho: Density of the sphere, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        X, Y, Z = self.vs.latent
        i, j, k = ijk

        d2 = (X - i) ** 2 + (Y - j) ** 2 + (Z - k) ** 2
        locs = d2 <= r**2

        if any(ypr):
            locs = self.vs.rotate(locs, ijk, ypr)

        self.lay(locs, rho, color)

    def make_cylinder(
        self,
        r: float,
        h: int,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        axis: VoxelAxes = VoxelAxes.z,
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a cylinder.

        Arguments:
            r: Radius of the cylinder.
            h: Height of the cylinder.
            ijk: Center coordinates (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
            axis: Orientation axis in {1, 2, 3}.
            rho: Density of the cylinder, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        X, Y, Z = self.vs.latent
        i, j, k = ijk

        match axis:
            case VoxelAxes.x:
                d2 = (Y - j) ** 2 + (Z - k) ** 2
                mask = (X >= i - h // 2) & (X < i + h // 2)
            case VoxelAxes.y:
                d2 = (X - i) ** 2 + (Z - k) ** 2
                mask = (Y >= j - h // 2) & (Y < j + h // 2)
            case VoxelAxes.z:
                d2 = (X - i) ** 2 + (Y - j) ** 2
                mask = (Z >= k - h // 2) & (Z < k + h // 2)
            case _:
                raise ValueError(
                    f'Expected an axis in {{1, 2, 3}}. Got: {axis}'
                )

        locs = (d2 <= r**2) & mask

        if any(ypr):
            locs = self.vs.rotate(locs, ijk, ypr)

        self.lay(locs, rho, color)

    def make_torus(
        self,
        R: float,
        r: float,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        axis: VoxelAxes = VoxelAxes.z,
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a torus.

        Arguments:
            R: Distance from center to tube center.
            r: Radius of the tube.
            ijk: Center coordinates (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
            axis: Orientation axis ('x', 'y', or 'z').
            rho: Density of the torus, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        X, Y, Z = self.vs.latent
        i, j, k = ijk

        match axis:
            case VoxelAxes.x:
                a, b, c = Y - j, Z - k, X - i
            case VoxelAxes.y:
                a, b, c = X - i, Z - k, Y - j
            case VoxelAxes.z:
                a, b, c = X - i, Y - j, Z - k
            case _:
                raise ValueError(
                    f'Expected an axis in {{1, 2, 3}}. Got: {axis}'
                )

        d2 = np.sqrt(a**2 + b**2) - R
        locs = (d2**2 + c**2) <= r**2

        if any(ypr):
            locs = self.vs.rotate(locs, ijk, ypr)

        self.lay(locs, rho, color)

    def make_cone(
        self,
        r: float,
        h: int,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        axis: VoxelAxes = VoxelAxes.z,
        invert: bool = False,
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a cone.

        Arguments:
            r: Base radius of the cone.
            h: Height of the cone.
            ijk: Base center coordinates (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
            axis: Orientation axis ('x', 'y', or 'z').
            invert: Invert the direction of the tip.
            rho: Density of the cone, default 1.0.
            color: The color.
        Side effects:
            Modifies `self.components', `self.component_rhos', and
            `self.component_cols'.
        """
        X, Y, Z = self.vs.latent
        i, j, k = ijk

        match axis:
            case VoxelAxes.x:
                a, b, c = Y - j, Z - k, X - i
                ijk_c = (
                    i + (h / 4) * (2 * invert - 1),
                    j,
                    k,
                )
            case VoxelAxes.y:
                a, b, c = X - i, Z - k, Y - j
                ijk_c = (
                    i,
                    j + (h / 4) * (2 * invert - 1),
                    k,
                )
            case VoxelAxes.z:
                a, b, c = X - i, Y - j, Z - k
                ijk_c = (
                    i,
                    j,
                    k + (h / 4) * (2 * invert - 1),
                )
            case _:
                raise ValueError(
                    f'Expected an axis in {{1, 2, 3}}. Got: {axis}'
                )

        lr = np.clip(r * (c / h) if invert else r * (1 - c / h), 0, r)

        d2 = a**2 + b**2
        locs = (d2 <= lr**2) & (c >= 0) & (c < h)

        if any(ypr):
            locs = self.vs.rotate(locs, ijk_c, ypr)

        self.lay(locs, rho, color)

    def make_rect_pyramid(
        self,
        lw: tuple,
        h: int,
        ijk: tuple = (0, 0, 0),
        ypr: tuple = (0, 0, 0),
        axis: VoxelAxes = VoxelAxes.z,
        invert: bool = False,
        rho: float = 1.0,
        color: str = 'g',
    ) -> None:
        """
        Create a rectangular pyramid.

        Arguments:
            lw: Length and width of the base.
            h: Height of the pyramid.
            ijk: Corner coordinates, determined by (0, 0, 0) |-> (i, j, k).
            ypr: Rotation angles in radians about (z, y, x) axes around center.
            axis: Orientation axis ('x', 'y', or 'z').
            invert: If True, creates an upside-down pyramid (point at bottom).
            rho: Density of the pyramid, default 1.0.
            color: The color.
        """
        X, Y, Z = self.vs.latent
        ll, w = lw
        i, j, k = ijk
        locs = np.zeros_like(X, dtype=bool)

        ijk_c = ijk
        for level in range(h):
            if invert:
                cur_w = w * (level + 1) / h
                cur_l = ll * (level + 1) / h
            else:
                cur_w = w * (h - level) / h
                cur_l = ll * (h - level) / h

            match axis:
                case VoxelAxes.x:
                    mask = (Y < cur_l) & (Z < cur_w)
                    x_pos = level if invert else (h - level - 1)
                    mask &= X == x_pos
                    shifted = self.vs.shift(mask, (x_pos + i, j, k))
                    ijk_c = (
                        i + (3 * h / 4) * (2 * invert - 1),
                        j + (w / 2),
                        k + (ll / 2),
                    )
                case VoxelAxes.y:
                    mask = (X < cur_l) & (Z < cur_w)
                    y_pos = level if invert else (h - level - 1)
                    mask &= Y == y_pos
                    shifted = self.vs.shift(mask, (i, y_pos + j, k))
                    ijk_c = (
                        i + (ll / 2),
                        j + (3 * h / 4) * (2 * invert - 1),
                        k + (w / 2),
                    )
                case VoxelAxes.z:
                    mask = (X < cur_l) & (Y < cur_w)
                    z_pos = level if invert else (h - level - 1)
                    mask &= Z == z_pos
                    shifted = self.vs.shift(mask, (i, j, z_pos + k))
                    ijk_c = (
                        i + (ll / 2),
                        j + (w / 2),
                        k + (3 * h / 4) * (2 * invert - 1),
                    )
                case _:
                    raise ValueError(
                        f'Expected an axis in {{1, 2, 3}}. Got: {axis}'
                    )

            locs |= shifted

        if any(ypr):
            locs = self.vs.rotate(locs, ijk_c, ypr)

        self.lay(locs, rho, color)

    def _reset_this(self) -> None:
        """
        Side effects:
            Clears `self.components' and resets `self.color'.
            For working in Jupyter notebooks.
        """
        self.components = self.default_empty.copy()
        self.component_rhos = self.default_empty.copy()
        self.component_cols = self.default_empty.copy()

    # Attributes are always restacked left to right.
    def __or__(self, o):
        if not isinstance(o, Voxel):
            raise TypeError('Operand must be a Voxel')
        if o.vs is not self.vs:
            raise ValueError(
                f'Voxels must share the same VoxelSpace. Got: {self.vs}, {o.vs}'
            )

        v = Voxel(self.vs)
        v._set_components(self.components + o.components)
        v._set_component_rhos(self.component_rhos + o.component_rhos)
        v._set_component_cols(self.component_cols + o.component_cols)
        return v
