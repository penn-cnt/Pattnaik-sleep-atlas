"""
    Code for making SurfIce plots by writing the surfice scripts to copy into the surfice command line.
"""
# imports
import os
import sys
from textwrap import dedent
from os.path import join as ospj
import warnings

import numpy as np

sys.path.append(ospj(os.path.dirname(__file__), '..', '..'))

from tools.config import Paths


# constants
# - surfice
with open(os.path.join(os.path.dirname(__file__), 'surfice_template.txt')) as file:
    surfice_script_outline = ''.join(file.readlines())

def plot_nodes(x, y, z, s, c, node_filename=None, mesh_filename=None, local=True, out_filename=None, return_script=True, node_thresh=None):
    """
        Create a surfice script for plotting a surface.
        Parameters:
            x, y, z: 1D arrays of coordinates
            c: 1D array of values to plot
            s: node size
            node_filename: filename for the .node file, if None, then make a temporary file, warn that it may be overwritten
            mesh_filename: file for the mesh, if None, then use default ICBM152 mesh
            out_filename: filename for the output .png file, if None, then don't save
            local: if True, then convert file paths to local paths e.g. /mnt/leif/littlab/users -> /Volumes/Users
            return_script: if True, then return the script as a string

        Returns:
            script: string for the surfice script
    """
    # get the number of points
    n = len(x)
    assert n == len(y) == len(z) == len(c) == len(s), 'x, y, z, c, and s must all be the same length'

    # make .node file
    if node_filename is None:
        node_filename = ospj(Paths.data_dir, 'surfice_plotting', 'node_files', "tmp.node")
        warnings.warn(f"node_filename is None, so using {node_filename}. This file may be overwritten.")

    else:
        node_filename = ospj(Paths.data_dir, 'surfice_plotting', 'node_files', f"{node_filename}.node")
    np.savetxt(
        fname=node_filename,
        X=np.c_[x, y, z, c, s],
        fmt="%.4d",
        delimiter=' ',
    )
    
    # set mesh filename if not given
    if mesh_filename is None:
        mesh_filename = 'BrainMesh_ICBM152.mz3'

    # make surfice script
    if local:
        mesh_filename = mesh_filename.replace('/mnt/leif/littlab/users', '/Volumes/Users')
        node_filename = node_filename.replace('/mnt/leif/littlab/users', '/Volumes/Users')
    if out_filename is not None:
        save_dir = ospj(Paths.fig_dir, 'surfice').replace('/mnt/leif/littlab/users', '/Volumes/Users')

        save_axial    = f"gl.viewaxial(1)\ngl.wait(200)\ngl.savebmp('{save_dir}/{out_filename}_axial.png')\ngl.wait(200)\ngl.viewaxial(0)"
        save_coronal  = f"gl.viewcoronal(1)\ngl.wait(200)\ngl.savebmp('{save_dir}/{out_filename}_coronal.png')\ngl.wait(200)\ngl.viewcoronal(0)"
        save_sagittal = f"gl.viewsagittal(1)\ngl.wait(200)\ngl.savebmp('{save_dir}/{out_filename}_sagittal.png')\ngl.wait(200)\ngl.viewsagittal(0)"

        # if they exist, then delete them by preprending os.remove
        save_axial = f"if os.path.exists('{ospj(save_dir, f'{out_filename}_axial.png')}'): os.remove('{ospj(save_dir, f'{out_filename}_axial.png')}')\n{save_axial}"
        save_coronal = f"if os.path.exists('{ospj(save_dir, f'{out_filename}_coronal.png')}'): os.remove('{ospj(save_dir, f'{out_filename}_coronal.png')}')\n{save_coronal}"
        save_sagittal = f"if os.path.exists('{ospj(save_dir, f'{out_filename}_sagittal.png')}'): os.remove('{ospj(save_dir, f'{out_filename}_sagittal.png')}')\n{save_sagittal}"
    else:
        save_axial    = ''
        save_coronal  = ''
        save_sagittal = ''

    if node_thresh is not None:
        node_thresh = f"gl.nodethresh({node_thresh[0]}, {node_thresh[1]})"
    else:
        node_thresh = ''

    script = surfice_script_outline.format(
        mesh_filename=mesh_filename,
        node_filename=node_filename,
        save_axial=save_axial,
        save_coronal=save_coronal,
        save_sagittal=save_sagittal,
        node_thresh=node_thresh,
    )

    # save script and return
    if out_filename is not None:
        with open(ospj(Paths.data_dir, 'surfice_plotting', 'scripts', f"{out_filename}.py"), 'w') as file:
            file.write(script)
    if return_script:
        return dedent(script)

# test code for main
if __name__ == '__main__':
    x = np.random.rand(100) * 100
    y = np.random.rand(100) * 100
    z = np.random.rand(100) * 100
    c = np.random.rand(100) * 100
    s = np.ones(100)
    out_filename = 'test_surfice_script'
    test = plot_nodes(x, y, z, s, c, out_filename=out_filename)
    print(test)