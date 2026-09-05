import math
import re

from batch_gcode_parser import NCParser

parser = NCParser()

lines = [
"N66 G1 Z169.625 F5000",
"N67 S200",
"N68 M3",
"N69 Z164.625 F14",
"N70 S1670",
"N71 M7",
"N72 G04 F2",
"N73 Z131.272 F67",
"N74 G04 F3",
"N75 M5",
"N76 G04 F2",
"N77 Z164.625 F2000",
"N78 Z169.625 F3000",
"N79 M9",
"N80 Z174.625 F5000",
"N81 G0 Z185 ",
"N82 G0 X44.519 ",
"N83 G0 Z174.625 ",
"N84 G1 Z169.625 F5000",
]

with open("test.mpf", "w") as f:
    f.write("\n".join(lines))

df = parser.parse_file("test.mpf")
print(df[['Block_ID', 'N_Number', 'Cmd_F', 'Is_G01', 'Delta_3D']])
