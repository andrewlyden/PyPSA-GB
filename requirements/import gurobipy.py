import os

os.environ["GRB_LICENSE_FILE"] = r"C:\Program Files\Gurobi\win64\bin\gurobi.lic"

import gurobipy

print(gurobipy.gurobi.getEnv()._getAttr_("LicenseType"))
