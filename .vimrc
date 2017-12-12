call plug#begin('~/.vim/plugged')

" Make sure you use single quotes
Plug 'junegunn/seoul256.vim'
Plug 'junegunn/vim-easy-align'

" Group dependencies, vim-snippets depends on ultisnips
"Plug 'SirVer/ultisnips' | Plug 'honza/vim-snippets' " requires vim 7.4

" On-demand loading
Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }
Plug 'tpope/vim-fireplace', { 'for': 'clojure' }

" Using git URL
Plug 'https://github.com/junegunn/vim-github-dashboard.git'

" Plugin options
Plug 'nsf/gocode', { 'tag': 'v.20150303', 'rtp': 'vim' }

" Plugin outside ~/.vim/plugged with post-update hook
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': 'yes \| ./install' }

" Unmanaged plugin (manually installed and updated)
Plug '~/my-prototype-plugin'

call plug#end()

" better colour scheme
source /afs/cern.ch/user/w/wfawcett/.vim/colour/cpp.vim

" Line indentation, see https://github.com/Yggdroot/indentLine
let g:indentLine_color_term = 239
"let g:indentLine_enabled = 0  " uncomment to disable
let g:indentLine_char = '.'


"""""""""""""""""""""""""""
" Marcos
"""""""""""""""""""""""""""
" c++
let @c = '^istd::cout << "" << std::endl;F"i'
" tex
let @f = '^i\begin{frame}{}xx10100100\end{frame}^[:s/xx10100100/\r\r^Mki'
let @t = '^i\begin{itemize}xx10100100\end{itemize}^[:s/xx10100100/\r\r^Mki  \item '
let @i = '^i\begin{frame}{}xx10100100\end{frame}^[:s/xx10100100/\r\r^Mk^i\begin{itemize}xx10100100\end{itemize}^[:s/xx10100100/\r\r^Mki  \item '


""""""""""""""""""""""""""""
" Skeletons
""""""""""""""""""""""""""""
au BufNewFile *.py 0r ~/.vim/skeletons/python

""""""""""""""""""""""""""""
" Vim user interface"
""""""""""""""""""""""""""""
" Start interactive EasyAlign in visual mode (e.g. vip<Enter>)
vmap <Enter> <Plug>(EasyAlign)

" Start interactive EasyAlign for a motion/text object (e.g. gaip)
nmap ga <Plug>(EasyAlign)

"Always show current position
set ruler

" Configure backspace so it acts as it should act
set backspace=eol,start,indent
set whichwrap+=<,>,h,l

" Ignore case when searching
set ignorecase

" When searching try to be smart about cases 
set smartcase

" Makes search act like search in modern browsers
set incsearch

" Highlight search results
set hlsearch

" Line numbering
"set number
set number relativenumber


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Text, tab and indent related
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Use spaces instead of tabs
set expandtab

" Be smart when using tabs ;)
set smarttab

" 1 tab == 2 spaces
set shiftwidth=2
set tabstop=2

au BufRead *.py setlocal shiftwidth=4
au BufRead *.py setlocal tabstop=4

"""""""""""""""""""""""""""""""""""""
" Spelling, text editing
"""""""""""""""""""""""""""""""""""""
set spelllang=en_gb

au BufRead *.tex setlocal spell
"au BufRead README setlocal spell
au BufRead *.txt setlocal tw=79
au BufRead *.tex setlocal tw=79
au BufRead README setlocal tw=120
au BufRead *.txt setlocal formatoptions+=t
au BufRead *.tex setlocal formatoptions+=t
au BufRead README setlocal formatoptions+=t


"remove errors from end } in c, otherwise always has errors for function lambdas
let c_no_curly_error=1
:map <F3> :nohl<CR>


" Tab completion
" will insert tab at beginning of line,
" will use completion if not at beginning
function! InsertTabWrapper()
  let col = col('.') - 1
  if !col || getline('.')[col - 1] !~ '\k'
    return "\<tab>"
  else
    return "\<c-p>"
  endif
endfunction



" Linebreak on 500 characters
set lbr
set tw=500

set ai "Auto indent
set si "Smart indent
set wrap "Wrap lines


""""""""""""""""""""""""""""
" Mappings
""""""""""""""""""""""""""""
" pastetoggle for better pasting
set pastetoggle=<F2>

" Mappings from http://vim.wikia.com/wiki/Swapping_characters,_words_and_lines#Mappings
" swap the current character with the next, without changing the cursor position:
nnoremap <silent> gc xph
" swap the current word with the next, without changing cursor position
nnoremap <silent> gw "_yiw:s/\(\%#\w\+\)\(\W\+\)\(\w\+\)/\3\2\1/<CR><c-o><c-l>
"swap the current word with the next, keeping cursor on current word: (This feels like pushing the word to the right.)
nnoremap <silent> gr "_yiw:s/\(\%#\w\+\)\(\_W\+\)\(\w\+\)/\3\2\1/<CR><c-o>/\w\+\_W\+<CR><c-l>
"swap the current paragraph with the next
nnoremap g{ {dap}p{
"swap word under the cursor to the right
nnoremap <silent> gh "_yiw?\w\+\_W\+\%#<CR>:s/\(\%#\w\+\)\(\_W\+\)\(\w\+\)/\3\2\1/<CR><c-o><c-l>

"inoremap <CL> <ESC>
"inoremap <CapsLock> <Esc>

" remap control-K to insert newline above current and go to end of line
inoremap <C-k> <Esc>O<Esc>jA
inoremap <F2> <C-p>
inoremap <F9> <C-p>

source /afs/cern.ch/user/w/wfawcett/.vim/plugin/comment.vim
"nnoremap <C-q> I#<Esc>
"inoremap <C-q> <Esc>:s/^/#<CR>
