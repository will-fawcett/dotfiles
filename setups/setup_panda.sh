#!/usr/bin/env bash

voms-proxy-init -voms atlas
source /afs/cern.ch/atlas/offline/external/GRID/ddm/DQ2Clients/setup.sh
source /afs/cern.ch/atlas/offline/external/GRID/DA/panda-client/latest/etc/panda/panda_setup.sh
export DQ2_LOCAL_SITE_ID=UKI-SOUTHGRID-OX-HEP_SCRATCHDISK
