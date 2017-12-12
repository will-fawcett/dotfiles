#! /bin/bash
#Macro to create grid proxy certificate and setup variables for panda data transfer

export X509_USER_PROXY=${HOME}/myProxy
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
voms-proxy-init -voms atlas -valid 96:00:00
localSetupDQ2Client --skipConfirm
localSetupPandaClient --noAthenaCheck
