# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# DO NOT USE THIS EVER
# Make python 2.7 the default
#scl enable python27 bash

 

# make sure up and down arrow searching is on
export INPUTRC=~/.inputrc

# prompt colouring
#PS1='\[\e[1;37m\][\u@\h \w]\$\[\e[0m\] '

################# History improvements ################

# increase .bash_hitory size
export HISTSIZE=100000

# ignore duplicates in history
export HISTCONTROL=ignoredups

# better format when using $ history
export HISTTIMEFORMAT='%F %T '

# path for atlas latex 
export PATH=/afs/cern.ch/sw/XML/texlive/latest/bin/x86_64-linux:$PATH


################# Other improvements ################

# autocorrect misspelt cd commands
shopt -s cdspell


#python paths
#export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/user/w/wfawcett/private/ServiceWork/ipin_monitoring/DCS_DB_query/moduleGroups



export RUCIO_ACCOUNT=wfawcett

# from software tutorial
#export ALRB_TutorialData=/afs/cern.ch/atlas/project/PAT/tutorial/triumf-sep2014
export ALRB_TutorialData=/afs/cern.ch/atlas/project/PAT/tutorial/cern-nov2014


# eos
export EOS_MGM_URL=root://eosatlas.cern.ch


[ -f ~/.fzf.bash ] && source ~/.fzf.bash

# added by Miniconda3 4.1.11 installer
#export PATH="/afs/cern.ch/user/w/wfawcett/private/deepLearningTutorial/miniconda3/bin:$PATH"

# Nice startup script
if [[ $- == *i* ]]; then
  source ~/setups/startup.sh
fi

