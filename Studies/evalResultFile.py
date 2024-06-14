import json
import os
import matplotlib.pyplot as plt


resultsType = "2d"
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Studies")
# runs used for presentation on 2d optim dataset: (uncomment only one of them)
fn = "results_2d_2023-07-01_15:13:49.json"
# fn = "results_2d_2023-07-01_15:26:48.json"

save_file = os.path.join(base_path, fn)

with open(save_file, "r") as dataFile:
    results = json.load(dataFile)

robots_ids = results["robot_ids"]
resolutions = results["resolutions"]
slices = results["slices"]
sample_sizes = results["sample_sizes"]

jointNumberByRobot = {}
for rId in robots_ids:
    jointNumberByRobot[rId] = results['data'][rId]['info']['n_joints']

max_cnts = {}

if resultsType == "3d":
    cov_by_sample_size_at_resolution = {}

    for resolution in resolutions:
        cov_by_sample_size_at_resolution[resolution] = {}
        for s in sample_sizes:
            cov_by_sample_size_at_resolution[resolution][s] = []
        for rob_id in robots_ids:
            resResults = results['data'][rob_id][resolution]
            for slc_nr in slices:
                for samp_size in sample_sizes:
                    cov_by_sample_size_at_resolution[resolution][samp_size].append(resResults[slc_nr]["fk"][samp_size]
                                                                                   ["coverage"])
    print(cov_by_sample_size_at_resolution)

elif resultsType == "2d":
    # by sample size
    points_by_resolution_and_sample_size = {}
    for res in resolutions:
        points_by_resolution_and_sample_size[res] = {}
        max_cnts[res] = []
        for s in sample_sizes:
            points_by_resolution_and_sample_size[res][s] = []
        for r in robots_ids:
            max_cnts[res].append(0)

    for idx, rob_id in enumerate(robots_ids):
        for resolution in resolutions:
            for sample_size in sample_sizes:
                dp = results["data"][rob_id][resolution]["fk"][sample_size]
                points_by_resolution_and_sample_size[resolution][sample_size].append((dp["cnt"], dp['time']))
                max_cnts[resolution][idx] = max(dp["cnt"], max_cnts[resolution][idx])
    for res in resolutions:
        for s_cnt in sample_sizes:
            total_cnt = sum([x[0] for x in points_by_resolution_and_sample_size[res][s_cnt]])
            total_time = sum([x[1] for x in points_by_resolution_and_sample_size[res][s_cnt]])
            avg_time = round(total_time / len(robots_ids), 2)
            print("---------------------------")
            print(f"Samples: {s_cnt}")
            print(f"\ttotal Cnt: {total_cnt}")
            print(f"\ttotal time: {round(total_time, 2)}s")
            print(f"\tavg time: {avg_time}s")
            perc_by_nJoints = {
                '0': [],
                '1': [],
                '2': [],
                '3': [],
                '4': [],
                '5': [],
                '6': [],
            }
            for rdx, rId in enumerate(robots_ids):
                points = points_by_resolution_and_sample_size[res][s_cnt][rdx][0]
                percOfMaxCnt = round(points / max_cnts[res][rdx] * 100, 2)
                perc_by_nJoints[str(jointNumberByRobot[rId])].append(percOfMaxCnt)
                # print(f"\t\tfor {jointNumberByRobot[rId]} {percOfMaxCnt}%")
            print("\tavg perc per joint:")
            for i in range(0, 7):
                if len(perc_by_nJoints[str(i)]) > 0:
                    print(f"\t\t{len(perc_by_nJoints[str(i)])} robots with {i} joints")
                    print(f"\t\tavg perc for {i} joints: "
                          f"{round(sum(perc_by_nJoints[str(i)])/len(perc_by_nJoints[str(i)]), 2)}%")
            print("\n")

    for res in resolutions:
        pltData = {
            "x": [],
            "h": []
        }
        for s_cnt in sample_sizes:
            pltData['x'].append(s_cnt)
            pltData['h'].append(sum([x[0] for x in points_by_resolution_and_sample_size[res][s_cnt]]))
        print(pltData)
        fig, ax = plt.subplots()
        ax.bar(pltData['x'], pltData['h'])
        ax.set_title(f"Accumulated datapoints at {res} by sample size")
        ax.set_xlabel("sample size")
        ax.set_ylabel("data points")
        plt.show()
