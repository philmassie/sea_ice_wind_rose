import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import datetime as dt
import math
pd.options.display.max_rows = 100

# Load up the data
ice = pd.read_csv("data/nsidc_global_nt_final_and_nrt.txt", skiprows=21)

# tidying up
ice.columns = [c.strip() for c in ice.columns]
ice[ice.columns[1:]] = ice[ice.columns[1:]].astype(float)
ice['date'] = ice["date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M"))
ice.sort_values("date", inplace=True)

# No date errors, so...
# add a couple of derived columns
ice["year"]=[i.year for i in ice["date"]]
ice["month"]=[i.strftime("%b") for i in ice["date"]]
ice["dayofyear"]=[i.dayofyear for i in ice["date"]]
# add a lazy-person-date date sort column
ice["sort_key"]=[float(str(i.year) + "." + str(i.dayofyear).zfill(3)) for i in ice["date"]]

# check for nulls
# So therea a month and a half of missing data - 1987-12-03 to 1988-01-12
ice.isnull().sum()
ice[ice.isnull().any(axis=1)]
ice.loc[(ice["sort_key"] > 1987.330) & (ice["sort_key"] < 1988.015),:][["extent", "area"]]

# Try interpolate - linear and 2-polynomial
ice["extent_poly"] = ice["extent"].interpolate("polynomial", order=2)
ice["extent_lin"] = ice["extent"].interpolate("linear")
ice["area_poly"] = ice["area"].interpolate("polynomial", order=2)
ice["area_lin"] = ice["area"].interpolate("linear")

# Check the different interpolations
"""
ice.loc[(ice["sort_key"] > 1987.330) & (ice["sort_key"] < 1988.015),:][["extent", "area", "extent_poly", "extent_lin", "area_poly", "area_lin"]]

pdat = ice.loc[(ice["sort_key"] > 1987.300) & (ice["sort_key"] < 1989.050),:]
y = pdat["extent_poly"]
y = pdat["extent_lin"]
y = pdat["area_poly"]
y = pdat["area_lin"]
sns.scatterplot(range(len(y)), y, linewidth=0, s=8)
plt.show()
plt.close()
"""

# Subset the columns we care most about - drop the 366th day of leap years to make sure all the years have the saame number of data points
ice_plt = ice.loc[ice["dayofyear"] <= 365, ["sort_key", "year", "month", "dayofyear", "extent_poly", "area_poly"]].sort_values("sort_key").reset_index(drop=True)

# Radians
#--------
# Radians column prep
max_angles = 365
# angles_base = np.linspace(0, 2*np.pi, max_angles, endpoint=False)
# angle_diff = np.diff(angles_base)[0]

# calculate 1 step (out of 365) in radians
rad_step = (2 * np.pi) / 365
# Data set starts at day 299, so that's our starting point in radians (zero based - first step is at o rad)
rad_start = rad_step * (ice_plt["dayofyear"][0] - 1)
# generate list of cumulative radians for each data point, an append to dataframe
# the polar plot wraps these efficiently so it seems a simple approach
rad_seq = [rad_start]
while len(rad_seq) < len(ice_plt):
    rad_seq.append(max(rad_seq) + rad_step)
ice_plt["rad_seq"] = rad_seq

# Colours
#--------
# prep the rgba colour arrays, 1 per year
colours_rgb = list(reversed(sns.color_palette("Spectral", len(ice["year"].unique()))))
colours_rgba = [np.append(np.array(col_tup), 1) for col_tup in colours_rgb]

pdf_colours = pd.DataFrame(
    zip(ice["year"].unique(), colours_rgba),
    columns=["year", "colour_map"]
)
ice_plt = ice_plt.merge(pdf_colours, on="year", how="left")

# test colours
"""
pdat = ice_plt.iloc[:150]
pdat = ice_plt.iloc[-150:]
sns.scatterplot(x=pdat.rad_seq, y=pdat.extent_poly, c=np.array(pdat.colour_map), linewidth=0, s=8)
plt.show()
"""

# prep the radial tick sequence
ext_max = ice_plt["extent_poly"].max()
rticks_upper = 5 * math.ceil(ext_max/5)
rticks_seq = list(range(5, rticks_upper, 5))

# Instantiate plot
try:
    plt.close()
except:
    pass
plt.rc('font', size=6)
fig=plt.figure(figsize=(6,6), dpi=300)
ax = fig.add_subplot(111, projection="polar")

# Restyle the gridlines
# ax.grid(linestyle='--') is supposed to work but for the life of me I couldnt get it to... so had to dig into the weeds a little
ygridlines = ax.yaxis.get_gridlines()
xgridlines = ax.xaxis.get_gridlines()

for gridline in ygridlines:
    gridline.set_linewidth(0.5)
    gridline.set_linestyle("--")

for gridline in xgridlines:
    gridline.set_linewidth(0.5)
    gridline.set_linestyle("--")

# Back to working formatting
pos=ax.get_rlabel_position()
ax.set_rlabel_position(90)
ax.set_rmax(30)
ax.set_rticks(rticks_seq)

ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.set_thetagrids(np.linspace(0, 2*np.pi, num=13) * 180/np.pi, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", " "])
fig.suptitle(r"Changing Global Sea Ice Extent, 1978-2021 (10$\mathregular{{^{{ 6 }} }}$ km$\mathregular{{^{{ 2 }} }}$)", fontsize=12)

# now the loop 'n save
plot_num = len(ice_plt)
for plot_num in range(len(ice_plt)):
    print(plot_num)
    # prep the plotting data
    plt_sub = ice_plt.iloc[:plot_num+1]
    
    # Labels, values and angles
    lbl_year = plt_sub["year"].max()
    plt.title(str(lbl_year),fontsize=10, y=-0.12)

    stats = np.array(plt_sub["extent_poly"])
    angles = np.array(plt_sub["rad_seq"])

    # colours - array of arrays
    colours = np.array(list(plt_sub["colour_map"]))

    # generate alphas for the first n (to 100) dots
    alphas = np.linspace(0.1, 1, (100 if len(angles) > 100 else len(angles)))
    if len(colours) == 1:
        alphas=[1]

    # pad alphas with 0.099, to len plot data
    if (len(angles) - len(alphas)) > 0:
        alphas = np.insert(alphas, 0, [0.099 for i in range(len(angles) - len(alphas))])

    # the fourth column needs to be your alphas
    colours[:, 3] = alphas

    # add the last background dot - woont remove this
    idx_bg = [i for i in range(len(colours)) if (colours[i][3] < 0.1)]
    if len(idx_bg) > 0:
        points_bg_plots = (angles[idx_bg[-1]], stats[idx_bg[-1]], colours[idx_bg[-1]])
        # add last background dot
        plt.plot(points_bg_plots[0], points_bg_plots[1], color=points_bg_plots[2], marker='o', markeredgewidth=0.0, markersize=3)

    # now add the foreground dots
    idx_fg = [i for i in range(len(colours)) if (colours[i][3] >= 0.1)]
    points_fg_plots = zip(angles[idx_fg], stats[idx_fg], colours[idx_fg])

    points_fg_tracker = []
    for point in points_fg_plots:
        points_fg_tracker.extend(
            plt.plot(point[0], point[1], color=point[2], marker='o', markeredgewidth=0.0, markersize=3)
        )

    # Save`
    ax.set_rmax(30)
    ax.set_rticks(rticks_seq)
    plt.savefig("plots_01/sea_ice_extent_" + str(plot_num).zfill(5) + ".jpg")

    for point in points_fg_tracker:
        point.remove()
    

plt.close()




"""
15000 frames
±20 sec

±724 FPS
=0.04pts
ffmpeg -i sea_ice_extent_%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p ../_out.mp4

ffmpeg -i sea_ice_extent_%05d.jpg -filter:v "setpts=0.05*PTS" -c:v libx264 -preset slow -pix_fmt yuv420p ../pts_0.05.mp4

ffmpeg -i sea_ice_extent_%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p  -filter:v "setpts=0.05*PTS" ../pts_0.05_2.mp4

ffmpeg -i plots_01/sea_ice_extent_%05d.jpg -filter:v "setpts=0.05*PTS" pts_0.05.mp4
"""