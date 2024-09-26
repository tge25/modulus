
import numpy as np
import os, sys
from dateutils import daterange, dateshift, makedate, splitdate, datetohrs, hrstodate
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import rcParams
import matplotlib.colors as mcolors
rcParams['patch.linewidth']=0.1

imember = 0

# --- read the data that NVIDIA colleagues provided.

infile = '/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v3_ice_0.nc'
nc = Dataset(infile, 'r')
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

for itime in range(0,24):
    for iforecast in range(1,2):
        fig = plt.figure(figsize=(20, 12))  # Adjusted figure size to accommodate two subplots 
        plt.style.use('dark_background')
        for plot in range(4):
            if plot==0:
                ax = plt.subplot(2,2,plot+1)  # First subplot
                output = np.log(nc.groups['prediction'].variables['precip'][imember,itime,iforecast,ymin:ymax, xmin:xmax])
                title = 'pred precip' 
                cmap = "magma"
                vmin=-3
                vmax=3
            elif plot==1:
                ax = plt.subplot(2,2,plot+1)  # First subplot
                output = np.log(nc.groups['truth'].variables['precip'][itime,iforecast,ymin:ymax, xmin:xmax])
                title = 'target precip' 
                cmap = "magma"
                vmin=-3
                vmax=3 
            elif plot==2:
                ax = plt.subplot(2,2,plot+1)  # First subplot
                cat_snow = nc.groups['prediction'].variables['cat_snow'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                cat_ice = nc.groups['prediction'].variables['cat_ice'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                cat_freez = nc.groups['prediction'].variables['cat_freez'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                cat_rain = nc.groups['prediction'].variables['cat_rain'][imember,itime,iforecast,ymin:ymax, xmin:xmax]
                cat_non = nc.groups['prediction'].variables['cat_non'][imember,itime,iforecast,ymin:ymax, xmin:xmax]                               
                output = np.zeros((ny, nx), dtype=int)
                title = 'pred precip type'
                for ix in range(nx):
                    for jy in range(ny):
                        x = np.array([cat_non[jy,ix], cat_snow[jy,ix], \
                            cat_rain[jy,ix], cat_freez[jy,ix], cat_ice[jy,ix]])
                        max_index = np.argmax(x)
                        output[jy,ix] = max_index
            elif plot==3:
                ax = plt.subplot(2,2,plot+1)  # Second subplot
                cat_snow = nc.groups['truth'].variables['cat_snow'][itime,iforecast,ymin:ymax, xmin:xmax]
                cat_ice = nc.groups['truth'].variables['cat_ice'][itime,iforecast,ymin:ymax, xmin:xmax]
                cat_freez = nc.groups['truth'].variables['cat_freez'][itime,iforecast,ymin:ymax, xmin:xmax]
                cat_rain = nc.groups['truth'].variables['cat_rain'][itime,iforecast,ymin:ymax, xmin:xmax]
                cat_non = nc.groups['truth'].variables['cat_non'][itime,iforecast,ymin:ymax, xmin:xmax]
                title = 'target precip type'
                output = np.zeros((ny, nx), dtype=int)
                for ix in range(nx):
                    for jy in range(ny):
                        x = np.array([cat_non[jy,ix], cat_snow[jy,ix], \
                            cat_rain[jy,ix], cat_freez[jy,ix], cat_ice[jy,ix]])
                        max_index = np.argmax(x)
                        output[jy,ix] = max_index
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
            
            if plot<2:  
                ax.set_title(title, fontsize=20, pad=10)
                CS2 = m.pcolormesh(x, y, output, \
                    cmap=cmap, vmin=vmin, vmax=vmax)
                m.drawcoastlines(linewidth=1,color='Gray')
                m.drawcountries(linewidth=1,color='Gray')
                m.drawstates(linewidth=0.4,color='Gray')
                m.drawcounties(linewidth=0.1,color='LightGray')
            elif plot==2:        
                ax.set_title(title, fontsize=20, pad=10)
                CS2 = m.contourf(x, y, output, clevels,\
                    cmap=None, colors=colorst, extend='both')
                m.drawcoastlines(linewidth=1,color='Gray')
                m.drawcountries(linewidth=1,color='Gray')
                m.drawstates(linewidth=0.4,color='Gray')
                m.drawcounties(linewidth=0.1,color='LightGray')
            elif plot==3:
                ax.set_title(title, fontsize=20, pad=10)
                CS2 = m.contourf(x, y, output, clevels,\
                    cmap=None, colors=colorst, extend='both')
                m.drawcoastlines(linewidth=1,color='Gray')
                m.drawcountries(linewidth=1,color='Gray')
                m.drawstates(linewidth=0.4,color='Gray')
                m.drawcounties(linewidth=0.1,color='LightGray')
            print(itime, iforecast, plot, flush=True)
        # ---- set plot title and save 
        plt.subplots_adjust(hspace=0.4)
        plt.suptitle(f"IC = {cyyyymmddhh} lead = {clead_hours:02d} h", fontsize=25, y=0.98)  # Adjust y to move title up
        # Adjust the layout to avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust rect to ensure suptitle and subplots do not overlap
        plot_title = './output_movie/'+cyyyymmddhh+f'{clead_hours:02d}.png'
        fig.savefig(plot_title, dpi=300)
        plt.close("all")
        print ('saving plot to file = ',plot_title)
        print ('Done!')

import imageio 
import os
writer = imageio.get_writer(f'precip_comp.mp4', fps = 2)
dirFiles = os.listdir('./output_movie/') # list of directory files
dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for im in dirFiles:
     writer.append_data(imageio.imread('./output_movie/'+im))
writer.close()
print("saved to twc_mvp1_movie.mp4")