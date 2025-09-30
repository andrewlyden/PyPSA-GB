for year in range(2010, 2020 + 1):
    csvfile = open(
        "../../data/renewables/atlite/outputs/PV/PV_" + str(year) + ".csv", "r"
    ).readlines()
    filename = 1
    chunks = 2500
    for i in range(len(csvfile)):
        if i % chunks == 0:
            open(
                "../../data/renewables/atlite/outputs/PV/PV_"
                + str(year)
                + "_"
                + str(filename)
                + ".csv",
                "w+",
            ).writelines(csvfile[i : i + chunks])
            filename += 1
