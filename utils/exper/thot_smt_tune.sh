# Author: Daniel Ortiz Mart\'inez
# *- bash -*

# Executes language and translation model tuning.

# \textbf{Categ}: modelling

########
print_desc()
{
    echo "thot_smt_tune written by Daniel Ortiz"
    echo "thot_smt_tune tunes the decoder parameters"
    echo "type \"thot_smt_tune --help\" to get usage information"
}

########
version()
{
    echo "thot_smt_tune is part of the thot package"
    echo "thot version "${version}
    echo "thot is GNU software written by Daniel Ortiz"
}

########
usage()
{
    echo "thot_smt_tune           [-pr <int>] -c <string>"
    echo "                        -s <string> -t <string> -o <string>"
    echo "                        [-qs <string>]"
    echo "                        -tdir <string> [-debug] [--help] [--version]"
    echo ""
    echo "-pr <int>               Number of processors (1 by default)"
    echo "-c <string>             Configuration file"
    echo "-s <string>             File with source sentences"
    echo "-t <string>             File with target sentences"
    echo "-o <string>             Output directory common to all processors."
    echo "                        NOTES:"
    echo "                         a) give absolute paths when using pbs clusters"
    echo "                         b) the directory should not exist"
    echo "-qs <string>            Specific options to be given to the qsub"
    echo "                        command (example: -qs \"-l pmem=1gb\")"
    echo "                        NOTE: ignore this if not using a PBS cluster"
    echo "-tdir <string>          Directory for temporary files. The directory should be"
    echo "                        accessible by all computation nodes in pbs clusters."
    echo "                        NOTES:"
    echo "                         a) give absolute paths when using pbs clusters"
    echo "                         b) ensure there is enough disk space in the partition"
    echo "-debug                  After ending, do not delete temporary files"
    echo "                        (for debugging purposes)"
    echo "--help                  Display this help and exit."
    echo "--version               Output version information and exit."
    echo ""
    echo "IMPORTANT WARNING: this utility does not yet work properly in pbs clusters"
}

########
get_absolute_path()
{
    file=$1
    dir=`$DIRNAME $file`
    if [ $dir = "." ]; then
        dir=""
    fi
    basefile=`$BASENAME $file`
    path=`$FIND $PWD/$dir -name ${basefile} 2>/dev/null`
    if [ -z "$path" ]; then
        path=$file
    fi
    echo $path
}

########
create_lm_files()
{
    # Check availability of lm files
    nlines=`ls ${lmfile}* 2>/dev/null | $WC -l`
    if [ $nlines -eq 0 ]; then
        echo "Error! language model files could not be found: ${lmfile}"
        exit 1
    fi

    # Create lm files
    for file in `ls ${lmfile}*`; do
        if [ $file = ${lmfile}.weights ]; then
            # Create regular file for the weights
            cp ${lmfile}.weights ${outd}/lm || { echo "Error while preparing language model files" >&2 ; exit 1; }
        else
            # Create hard links for the rest of the files
            $LN -f $file ${outd}/lm || { echo "Error while preparing language model files" >&2 ; exit 1; }
        fi
#        cp $file ${outd}/lm || { echo "Error while preparing language model files" >&2 ; exit 1; }
    done
}


########
lm_downhill()
{
    # Export required variables
    export LM=$newlmfile
    export TEST=$tcorpus
    export ORDER=`cat $lmfile.weights | $AWK '{printf"%d",$1}'`
    export NUMBUCK=`cat $lmfile.weights | $AWK '{printf"%d",$2}'`
    export BUCKSIZE=`cat $lmfile.weights | $AWK '{printf"%d",$3}'`

    # Generate information for weight initialisation
va_opt=`${bindir}/thot_gen_init_file_with_jmlm_weights ${ORDER} ${NUMBUCK} ${BUCKSIZE} -0 | $AWK '{for(i=4;i<=NF;++i) printf"%s ",$i}'`
iv_opt=`${bindir}/thot_gen_init_file_with_jmlm_weights ${ORDER} ${NUMBUCK} ${BUCKSIZE} 0.5 | $AWK '{for(i=4;i<=NF;++i) printf"%s ",$i}'`

    # Execute tuning algorithm
${bindir}/thot_dhs_min -tdir $tdir -va ${va_opt} -iv ${iv_opt} \
-ftol $ftol -o ${outd}/lm_adjw -u ${bindir}/thot_dhs_trgfunc_jmlm || exit 1
}

########
tune_lm()
{
    # Obtain path of lm file
    lmfile=`$GREP "\-lm " $cmdline_cfg | $AWK '{printf"%s",$2}'`
    baselmfile=`basename $lmfile`

    # Create directory for lm files
    if [ -d ${outd}/lm ]; then
        echo "Warning! directory for language model does exist" >&2 
    else
        mkdir -p ${outd}/lm || { echo "Error! cannot create directory for language model" >&2; exit 1; }
    fi
    
    # Create initial lm files
    create_lm_files

    # Obtain new lm file name
    newlmfile=${outd}/lm/${baselmfile}

    # Tune language model
    lm_downhill
}

########
create_tm_dev_files()
{
    # Check availability of tm_dev files
    nlines=`ls ${tmfile}* 2>/dev/null | $WC -l`
    if [ $nlines -eq 0 ]; then
        echo "Error! translation model files could not be found: ${tmfile}"
        exit 1
    fi

    # Create tm files
    for file in `ls ${tmfile}*`; do
        if [ $file = ${tmfile}.ttable ]; then
            # Create regular file for the weights
            cp ${tmfile}.ttable ${outd}/tm_dev/${basetmfile}.ttable || { echo "Error while preparing translation model files" >&2 ; exit 1; }
        else
            # Create hard links for the rest of the files
            $LN -f $file ${outd}/tm_dev || { echo "Error while preparing translation model files" >&2 ; exit 1; }
        fi
    done
}

########
create_tm_files()
{
    # Check availability of tm_dev files
    nlines=`ls ${tmfile}* 2>/dev/null | $WC -l`
    if [ $nlines -eq 0 ]; then
        echo "Error! translation model files could not be found: ${tmfile}"
        exit 1
    fi

    # Create tm files
    for file in `ls ${tmfile}*`; do
            # Create hard links for each file
            $LN -f $file ${outd}/tm || { echo "Error while preparing translation model files" >&2 ; exit 1; }
    done
}

########
filter_ttable()
{
${bindir}/thot_filter_ttable -t ${tmfile}.ttable \
        -c $scorpus -n 20 -T $tdir > ${outd}/tm_dev/${basetmfile}.ttable 2> ${outd}/tm_dev/${basetmfile}.ttable.log
}

########
create_cfg_file_for_tuning()
{
    cat $cmdline_cfg | $AWK -v nlm=$newlmfile -v ntm=$newtmdevfile \
                         '{
                           if($1=="-lm") $2=nlm
                           if($1=="-tm") $2=ntm
                           printf"%s\n",$0
                          }'
}

########
obtain_loglin_nonneg_const()
{
    echo `$PHRDECODER --config 2>&1 | $GREP "\- Weights" | $AWK -F , '{for(i=1;i<=NF;++i) {if(i==1 || i==3) printf"0 "; else printf"1 "}}'`
}

########
obtain_loglin_va_opt_values()
{
    echo `$PHRDECODER --config 2>&1 | $GREP "\- Weights" | $AWK -F , '{for(i=1;i<=NF;++i) printf"-0 "}'`
}

########
obtain_loglin_iv_opt_values()
{
    echo `$PHRDECODER --config 2>&1 | $GREP "\- Weights" | $AWK -F , '{for(i=1;i<=NF;++i) printf"1 "}'`
}

########
loglin_downhill()
{
    # Export required variables
    export CFGFILE=${outd}/tune_loglin.cfg
    export TEST=$scorpus
    export REF=$tcorpus
export PHRDECODER=${bindir}/thot_decoder
# export PHRDECODER=${bindir}/thot_pbs_dec
# export ADD_DEC_OPTIONS="-pr ${pr_val} -sdir $tdir"
# export QS="${qs_par}"
    export MEASURE="BLEU"
    export USE_NBEST_OPT=0

    # Generate information for weight initialisation
    export NON_NEG_CONST=`obtain_loglin_nonneg_const`
    va_opt=`obtain_loglin_va_opt_values`
    iv_opt=`obtain_loglin_iv_opt_values`

    # Execute tuning algorithm
${bindir}/thot_dhs_min -tdir $tdir -va ${va_opt} -iv ${iv_opt} \
-ftol $ftol -o ${outd}/tm_adjw -u ${bindir}/thot_dhs_smt_trgfunc || exit 1
}

########
create_cfg_file_for_tuned_sys()
{
    # Obtain log-linear weights
    tmweights=`cat ${outd}/tm_adjw.out`

    # Print data regarding development files
    echo "# [SCRIPT_INFO] tool: thot_smt_tune"
    echo "# [SCRIPT_INFO] source dev. file: $scorpus" 
    echo "# [SCRIPT_INFO] target dev. file: $tcorpus" 
    echo "# [SCRIPT_INFO]"

    # Create file from command line file
    cat ${outd}/tune_loglin.cfg | $SED s'@/tm_dev/@/tm/@'| \
        $AWK -v tmweights="$tmweights" \
                            '{
                               if($1=="#" && $2=="-tmw")
                               {
                                 printf"-tmw %s\n",tmweights
                               }
                               else printf "%s\n",$0
                             }'
}

########
tune_loglin()
{
    # Obtain path of lm file
    lmfile=`$GREP "\-lm " $cmdline_cfg | $AWK '{printf"%s",$2}'`
    baselmfile=`basename $lmfile`

    # Create directory for lm files
    if [ -d ${outd}/lm ]; then
        lm_dir_already_exist=1
    else
        mkdir -p ${outd}/lm || { echo "Error! cannot create directory for language model" >&2; exit 1; }
        lm_dir_already_exist=0
    fi

    if [ ${lm_dir_already_exist} -eq 0 ]; then
        # Create initial lm files
        create_lm_files
    fi

    # Obtain new lm file name
    newlmfile=${outd}/lm/${baselmfile}

    ######

    # Obtain path of tm file
    tmfile=`$GREP "\-tm " $cmdline_cfg | $AWK '{printf"%s",$2}'`
    basetmfile=`basename $tmfile`

    # Create directory for tm files for development corpus
    if [ -d ${outd}/tm_dev ]; then
        echo "Warning! directory for dev. translation model does exist" >&2 
    else
        mkdir -p ${outd}/tm_dev || { echo "Error! cannot create directory for translation model" >&2; exit 1; }
    fi

    # Create initial tm_dev files
    create_tm_dev_files

    # Obtain new tm file name for development corpus
    newtmdevfile=${outd}/tm_dev/${basetmfile}

    # Filter translation table
    filter_ttable

    # Create cfg file for tuning
    create_cfg_file_for_tuning > ${outd}/tune_loglin.cfg

    # Tune log-linear model
    loglin_downhill

    ######

    # Create directory for tm files
    if [ -d ${outd}/tm ]; then
        echo "Warning! directory for translation model does exist" >&2 
    else
        mkdir -p ${outd}/tm || { echo "Error! cannot create directory for translation model" >&2; exit 1; }
    fi

    # Create initial tm files
    create_tm_files

    # Create cfg file of tuned system
    create_cfg_file_for_tuned_sys > ${outd}/tuned_for_dev.cfg
}

########
if [ $# -lt 1 ]; then
    print_desc
    exit 1
fi

# Read parameters
pr_given=0
pr_val=1
c_given=0
s_given=0
t_given=0
o_given=0
qs_given=0
unk_given=0
tdir_given=0
debug=0

while [ $# -ne 0 ]; do
    case $1 in
        "--help") usage
            exit 0
            ;;
        "--version") version
            exit 0
            ;;
        "-pr") shift
            if [ $# -ne 0 ]; then
                pr_val=$1
                pr_given=1
            fi
            ;;
        "-c") shift
            if [ $# -ne 0 ]; then
                cmdline_cfg=$1
                c_given=1
            fi
            ;;
        "-s") shift
            if [ $# -ne 0 ]; then
                scorpus=$1
                s_given=1
            fi
            ;;
        "-t") shift
            if [ $# -ne 0 ]; then
                tcorpus=$1
                t_given=1
            fi
            ;;
        "-o") shift
            if [ $# -ne 0 ]; then
                outd=$1
                o_given=1
            fi
            ;;
        "-qs") shift
            if [ $# -ne 0 ]; then
                qs_opt="-qs"
                qs_par=$1
                qs_given=1
            else
                qs_given=0
            fi
            ;;
        "-unk") unk_given=1
            unk_opt="-unk"
            ;;
        "-tdir") shift
            if [ $# -ne 0 ]; then
                tdir="$1"
                tdir_given=1
            fi
            ;;
        "-sdir") shift
            if [ $# -ne 0 ]; then
                sdir_opt="-sdir $1"
                sdir_given=1
            fi
            ;;
        "-debug") debug=1
            debug_opt="-debug"
            ;;
    esac
    shift
done

# Check parameters
if [ ${c_given} -eq 0 ]; then
    echo "Error! -cfg parameter not given" >&2
    exit 1
else
    if [ ! -f ${cmdline_cfg} ]; then
        echo "Error! file ${cmdline_cfg} does not exist" >&2
        exit 1
    else
        # Obtain absolute path
        cmdline_cfg=`get_absolute_path ${cmdline_cfg}`
    fi
fi

if [ ${s_given} -eq 0 ]; then
    echo "Error! -s parameter not given!" >&2
    exit 1
else
    if [ ! -f ${scorpus} ]; then
        echo "Error! file ${scorpus} does not exist" >&2
        exit 1
    else
        # Obtain absolute path
        scorpus=`get_absolute_path $scorpus`
    fi
fi

if [ ${t_given} -eq 0 ]; then        
    echo "Error! -t parameter not given!" >&2
    exit 1
else
    if [ ! -f ${tcorpus} ]; then
        echo "Error! file ${tcorpus} does not exist" >&2
        exit 1
    else
        # Obtain absolute path
        tcorpus=`get_absolute_path $tcorpus`
    fi
fi

if [ ${o_given} -eq 0 ]; then
    echo "Error! -o parameter not given!" >&2
    exit 1
else
    if [ -d ${outd} ]; then
        echo "Warning! output directory does exist" >&2 
        # echo "Error! output directory should not exist" >&2 
        # exit 1
    else
        mkdir -p ${outd} || { echo "Error! cannot create output directory" >&2; exit 1; }
    fi
    # Obtain absolute path
    outd=`get_absolute_path $outd`
fi

if [ ${tdir_given} -eq 0 ]; then
    echo "Error! -tdir parameter not given" >&2
    exit 1
else
    if [ ! -d ${tdir} ]; then
        echo "Error! directory ${tdir} does not exist" >&2
        exit 1            
    fi
fi

# Set default parameters
ftol=0.001

# Tune models
tune_lm
tune_loglin