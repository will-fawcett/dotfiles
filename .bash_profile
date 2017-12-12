# .bash_profile

# From software tutorial
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase

# For quick SCP
export lx='fawcett@pplxint8.physics.ox.ac.uk'

# Load aliases
source ~/.bash_alias



# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/bin
export PATH


tbrowser () {
  # Check a file has been specified
  if (( $# == 0 )); then
    echo "No file(s) specified."
  else
    # For each file, check it exists
    for i; do
      if [ ! -f $i ]; then
        echo "Could not find file $i"
        return 1;
      fi
    done
    root -l $* $HOME/.macros/newBrowser.C
  fi
}
