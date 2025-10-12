import urdf2mjcf

urdf = "SO101/so101_new_calib.urdf"
outpath = "SO101/so101.mjcf"

c = urdf2mjcf.convert
c.convert_urdf_to_mjcf(urdf, outpath)
