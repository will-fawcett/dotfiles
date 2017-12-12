
#
curr_dir=`pwd`

cd /home/fawcett/scripts/latex/rundir
module load texlive

pdflatex "\def\tablefile{$curr_dir/$1} \input{../make_table.tex}"
pdflatex "\def\tablefile{$curr_dir/$1} \input{../make_table.tex}"

convert -density 300 make_table.pdf -quality 100 make_table.png

mv make_table.pdf $curr_dir/${1%.tex}.pdf
mv make_table.png $curr_dir/${1%.tex}.png
