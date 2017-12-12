
if [ -z ${MJDATADIR+x} ]; then 
    cd /data/atlas/atlasdata/kalderon/Multijets/MultijetAnalysis13TeV/trunk
    source setup_environment.sh
fi

pyv="$(python -V 2>&1)"
shortver=${pyv:9:1}

if [[ $shortver -lt 7 ]]; then
    echo $pyv
    module load python/2.7
fi

python /data/atlas/atlasdata/kalderon/Multijets/MultijetAnalysis13TeV/trunk/python/monitoring/check_output.py $1
