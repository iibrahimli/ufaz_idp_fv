#! /bin/bash

# creates an archive in the same directory
# as the given folder. 
function compress {
    data_dir="$(readlink -f "${1}")"
    data_dir_name="$(basename "${data_dir}")"
    archive="$(dirname "${data_dir}")/${data_dir_name}.tar.gz"

    echo "Compressing $data_dir to $archive ..."
    tar -cjf "$archive" -C "${data_dir}/.." "$data_dir_name"  && echo "Done"
}


if [ $# -ne 1 ]; then
    echo "Invalid number of arguments"
    exit 1
fi

compress "$1"