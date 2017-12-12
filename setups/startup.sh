#! /bin/bash

shopt -s expand_aliases

str=`cat /etc/system-release | grep release`
arr=(${str// / })
num=${arr[3]}
vs=(${num//./ })
ver=${vs[0]}
#echo "Detected using SLC$ver"

oldPS1="[\u@\h \W]\$ "

export isOxford=False
export isGeneva=False
export isLXplus=False

if [ ${HOSTNAME:0:7} = "pplxint" ]; then
  export isOxford=True
  echo "Hello $USER at Oxford"
fi
if [ ${HOSTNAME:0:6} = "lxplus" ]; then
  export isLXplus=True
  echo "Hello $USER at LXplus"
  source ~/.bash_alias_lxplus
fi
if [ "${HOSTNAME:0:5}" = "atlas" ]; then
  export isGeneva=True
  echo "Hello $USER at Geneva"
  source ~/.bash_alias_geneva
fi

# colours for bash prompt
white=$(tput setaf 7)
red=$(tput setaf 1)
green=$(tput setaf 2)
blue=$(tput setaf 4)
reset=$(tput sgr0)

if [ $isGeneva == "True" ] || [ $isLXplus == "True" ] # ie on LXPlus or Geneva 
    then
    echo
    echo -e "\e[00;31mChoose an environment:\e[00m"
    echo -e "\e[00;36m1 - tkLayout"
    echo -e "2 - FCCSW"
    echo -e "3 - ACTS"
    echo -e "4 - Delphes\e[00m"
    echo "else - clean"
    echo
        
    echo -n "Choose: > "
    read var

  if [[ $var == "1" ]]; then
    echo "Setting up tkLayout"
    PS1="\[$blue\][\h@tkLayout \W]\[$reset\]\$ "

    if [[ $isLXplus == "True" ]]; then
      echo "test"
      cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc
      source setup_environment.sh
      cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc/tkLayout
      source setup_slc6.sh

      # make python work 
      export PYTHONDIR=/afs/cern.ch/sw/lcg/external/Python/2.7.4/x86_64-slc6-gcc48-opt
      export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/sw/lcg/app/releases/ROOT/6.04.18/x86_64-slc6-gcc49-opt/root/lib
      export LD_LIBRARY_PATH=/afs/cern.ch/sw/lcg/app/releases/ROOT/6.04.18/x86_64-slc6-gcc49-opt/root/lib:/lib:/afs/cern.ch/sw/lcg/contrib/gcc/4.9/x86_64-slc6-gcc49-opt/lib64:/afs/cern.ch/sw/lcg/app/releases/ROOT/6.04.18/x86_64-slc6-gcc49-opt/root/lib:/opt/rh/python27/root/usr/lib64
    elif [[ $isGeneva == "True" ]]; then 
      echo "tkLayout not yet setup in Geneva"
    fi

  elif [[ $var == "2" ]]
    then
    echo "Setting up FCCSW"
    PS1="\[$blue\][\h@FCCSW \W]\[$reset\]\$ "
    cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc
    source setup_environment.sh
    cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc/FCCSW 
    source init.sh 

  elif [[ $var == "3" ]]
    then
    echo "Setting up ACTS framework"
    cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc
    source setup_environment.sh
    #export PS1="\[\e[0;33m[\h@ACTS \W]\$ \e[m\]\[\]" # orange
    #cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc/acts
    cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc/acts/acts-framework
    #source setup.sh
    PS1="\[$red\][\h@ACTS \W]\[$reset\]\$ "

  elif [[ $var == "4" ]]; then
    echo "Setting up Delphes (+Pythia8)"

    PS1="\[$white\][\h@delphes \W]\[$reset\]\$ "

    # Machine specific path
    if [[ $isLXplus == "True" ]]; then
      cd /afs/cern.ch/user/w/wfawcett/private/geneva/fcc
    elif [[ $isGeneva == "True" ]]; then
      cd /atlas/users/wfawcett/fcc
    fi
  
    # Common relative paths 
    source setup_environment.sh
    cd delphes
    source setup.sh
      
  else
    echo "Starting in clean environment"
  fi
fi
