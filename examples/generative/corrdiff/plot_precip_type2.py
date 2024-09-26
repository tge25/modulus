import scipy.io
import numpy as np
import os, sys
import csv
from dateutils import daterange, dateshift, makedate, splitdate, datetohrs, hrstodate
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import rcParams
import matplotlib.colors as mcolors
rcParams['patch.linewidth']=0.1

imember = 0

# --- read the data that NVIDIA colleagues provided.

infile = '/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_diffusion_v3_movie_winter_storms2_0.nc'
nc = Dataset(infile, 'r')
    #7 different; units = "hours since 1990-01-01 00:00:00"  IC?
lon = nc.variables['lon'][:,:]
lon = lon-360.
lat = nc.variables['lat'][:,:]
ymin = 0
ymax = -1
xmin = 0
xmax = -1

lat = lat[ymin:ymax, xmin:xmax]
lon = lon[ymin:ymax, xmin:xmax]

ny, nx = np.shape(lon)

for itime in range(5):
    for iforecast in range(0,9):
        fig = plt.figure(figsize=(24, 15))  # Adjusted figure size to accommodate two subplots 
        for plot in range(9):
            if plot==0:   
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['input'].variables['u10m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'input u10m'  
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==1:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['prediction'].variables['u10m'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'pred u10m'  
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==2:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['truth'].variables['u10m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'target u10m' 
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==3:   
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['input'].variables['v10m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'input v10m'  
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==4:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['prediction'].variables['v10m'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'pred v10m'  
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==5:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['truth'].variables['v10m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'target v10m' 
                cmap = 'RdBu'
                vmin=-17
                vmax=17
            elif plot==6:   
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['input'].variables['t2m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'input t2m'  
                cmap = 'coolwarm'
                vmin=250
                vmax=310
            elif plot==7:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['prediction'].variables['t2m'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'pred t2m'  
                cmap = 'coolwarm'
                vmin=250
                vmax=310
            elif plot==8:
                ax = plt.subplot(3,3,plot+1)  # First subplot
                output = nc.groups['truth'].variables['t2m'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'target t2m' 
                cmap = 'coolwarm'
                vmin=250
                vmax=310

            time = nc.variables['time'][itime] 

            # --- get the year, month, day, hour associated with time.

            hrs_0_to_1990 = datetohrs('1990010100',mixedcal=True) 
                # 1990010100, hrs since day 1 CE.
            hrs_0_to_input = hrs_0_to_1990 + time.astype(int)
            cyyyymmddhh = hrstodate(hrs_0_to_input)
            iyyyy,imm,idd,ihh = splitdate(cyyyymmddhh)
            cmonths = ['Jan','Feb','Mar','Apr','May','Jun',\
                'Jul','Aug','Sep','Oct','Nov','Dec'] 
            clead_hours = iforecast*3
            cyyyymmddhh_forecast = dateshift(cyyyymmddhh, int(clead_hours))
                
            m = Basemap(rsphere=(6378137.00,6356752.3142),\
                resolution='l',area_thresh=1000.,projection='lcc',\
                lat_1=38.5,lat_2=38.5,lat_0=lat[ny//2,nx//2],lon_0=262.5-360.,\
                llcrnrlon=lon[0,0],llcrnrlat=lat[0,0],urcrnrlon=lon[-1,-1],\
                urcrnrlat=lat[-1,-1])

            x, y = m(lon, lat)
            clevels = [-0.001,0.999,1.999,2.999,3.999,4.999]
            colorst = ['Black','LightSkyBlue','LightGreen', 'LightCoral', 'Yellow']
            clabels = ['    None','    Snow','    Rain','    Freezing', "Ice pellet"]
            
            ax.set_title(title, fontsize=20, pad=10)
            CS2 = m.pcolormesh(x, y, output, \
                cmap=cmap, vmin=vmin, vmax=vmax)
            m.drawcoastlines(linewidth=1,color='Black')
            m.drawcountries(linewidth=1,color='Black')
            m.drawstates(linewidth=0.4,color='Black')
            m.drawcounties(linewidth=0.1,color='Black')
            print(itime, iforecast, plot, flush=True)
        # ---- set plot title and save 
        plt.subplots_adjust(hspace=0.4)
        plt.suptitle(f"IC = {cyyyymmddhh} lead = {clead_hours:02d} h", fontsize=30, y=0.98)  # Adjust y to move title up
        # Adjust the layout to avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust rect to ensure suptitle and subplots do not overlap
        plot_title = './output_movie/'+cyyyymmddhh+f'{clead_hours:02d}.png'
        fig.savefig(plot_title, dpi=300)
        plt.close('all')
        print ('saving plot to file = ',plot_title)
        print ('Done!')

import imageio 
import os
writer = imageio.get_writer(f'general_fields_comp_west.mp4', fps = 2)
dirFiles = os.listdir('./output_movie/') # list of directory files
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for im in dirFiles:
     writer.append_data(imageio.imread('./output_movie/'+im))
writer.close()
print("saved to twc_mvp1_movie.mp4")