for FILE in ../time_sel/6hrly_wrf2d_d01_ctrl_U10_20*.nc
do
    cdo remapnn,target.txt $FILE regrid_10_$(basename -- $FILE)
done
