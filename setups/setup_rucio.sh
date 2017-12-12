#! /bin/bash

source ~/setups/get_proxy.sh

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
shopt -s expand_aliases
export DQ2_LOCAL_SITE_ID='UKI-SOUTHGRID-OX-HEP_LOCALGROUPDISK'
export RUCIO_ACCOUNT='wfawcett'

if [[ "$X509_USER_PROXY" == "/home/fawcett/myProxy" ]]; then
    echo $X509_USER_PROXY
else
    echo $X509_USER_PROXY
    cp -u $X509_USER_PROXY /home/fawcett/myProxy
    export X509_USER_PROXY='/home/fawcett/myProxy'
    echo $X509_USER_PROXY
fi

setupATLAS
localSetupRucioClients


