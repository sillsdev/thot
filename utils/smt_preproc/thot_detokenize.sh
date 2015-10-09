# Author: Daniel Ortiz Mart\'inez
# *- bash -*

########
if [ $# -lt 1 ]; then
    echo "thot_detokenize -f <string> -r <string> [-t <string>]"
    echo ""
    echo "-f <string>    File with text to be detokenized (can be read from standard"
    echo "               input)"
    echo "-r <string>    File with raw text in the language of interest"
    echo "-t <string>    File with tokenized version of the raw text using"    
    echo "               an arbitrary tokenizer. If not given, \"thot_tokenize\" is"
    echo "               used internally"
else
    
    # Read parameters
    f_given=0
    r_given=0
    t_given=0
    while [ $# -ne 0 ]; do
        case $1 in
        "-f") shift
            if [ $# -ne 0 ]; then
                ffile=$1
                f_given=1
            fi
            ;;
        "-r") shift
            if [ $# -ne 0 ]; then
                rfile=$1
                r_given=1
            fi
            ;;
        "-t") shift
            if [ $# -ne 0 ]; then
                tfile=$1
                t_given=1
                topt="-t $tfile"
            fi
            ;;
        esac
        shift
    done

    # Check parameters
    if [ ${f_given} -eq 0 ]; then
        echo "Error! -f parameter not given" >&2
        exit 1
    fi

    if [ ! -f ${ffile} ]; then
        echo "Error! ${ffile} file does not exist" >&2
        exit 1
    fi

    if [ ${r_given} -eq 0 ]; then
        echo "Error! -r parameter not given" >&2
        exit 1
    fi

    if [ ! -f ${rfile} ]; then
        echo "Error! ${rfile} file does not exist" >&2
        exit 1
    fi

    if [ ${t_given} -eq 1 -a ! -f ${rfile} ]; then
        echo "Error! ${tfile} file does not exist" >&2
    fi

    ## Process parameters
    
    # Create directory for temporary files
    TMPDIR=`mktemp -d`
    trap "rm -rf $TMPDIR 2>/dev/null" EXIT

    # Obtain subset of raw file (this is done to speed up computations)
    maxfsize=500000
${bindir}/thot_shuffle 31415 ${rfile} > $TMPDIR/rf_shuff
    head -n ${maxfsize} $TMPDIR/rf_shuff > $TMPDIR/rf_shuff.trunc

    # Train models
    $bindir/thot_train_detok_model -r $TMPDIR/rf_shuff.trunc ${topt} -o $TMPDIR/models

    # Tune weights
    # TBD

    # Detokenize sentences
    $bindir/thot_detok_translator -f ${ffile} -m $TMPDIR/models -w "1 0 0 1" 

fi