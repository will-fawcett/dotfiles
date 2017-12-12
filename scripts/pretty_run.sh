#run $1 $2 $3 $4 $5 2>&1 | grep -E --color=always 'error|Error|ERROR|$' | GREP_COLOR='01;36' grep -E --color=auto 'warning|Warning|WARNING|$'
