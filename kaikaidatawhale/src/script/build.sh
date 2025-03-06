#!/bin/bash

# 获取脚本的绝对路径
SCRIPT_PATH="$(realpath "$0")"  

# 获取脚本所在的目录
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"  

build_dir="$SCRIPT_DIR/../build"

function kaikai() {
echo " /***                                                  "
echo " *  /$$                 /$$ /$$                 /$$    "
echo " * | $$                |__/| $$                |__/    "
echo " * | $$   /$$  /$$$$$$  /$$| $$   /$$  /$$$$$$  /$$    "
echo " * | $$  /$$/ |____  $$| $$| $$  /$$/ |____  $$| $$    "
echo " * | $$$$$$/   /$$$$$$$| $$| $$$$$$/   /$$$$$$$| $$    "
echo " * | $$_  $$  /$$__  $$| $$| $$_  $$  /$$__  $$| $$    "
echo " * | $$ \  $$|  $$$$$$$| $$| $$ \  $$|  $$$$$$$| $$    "
echo " * |__/  \__/ \_______/|__/|__/  \__/ \_______/|__/    "
echo "                                                       "
echo "  "

}

# 定义帮助信息
function show_help() {
    kaikai
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help        Display help message"
    echo "  -c, --clean       Clean the workspace"
    echo "  -t, --test        Run tests for a specific program (e.g., -t 1)."  
    echo "  -a, --all         Build all versions"
}

# 定义测试函数
function run_test() {
    local project_id="$1"
    if [ -z "$project_id" ]; then
        echo "Please specify the project name"
        show_help
        exit 1
    fi

    if ! [[ "$project_id" =~ ^[0-9]+$ ]]; then
        echo "Invalid project id: $project_id"
        exit 1
    fi

    echo "Running tests for project $project_id"

    case  "$project_id" in 
        1)
            echo "Running tests for course 1"
            build_dir="$SCRIPT_DIR/../build/1-course"
            echo "Build directory: $build_dir"
            cmake -B "$build_dir/" -S "$build_dir/../../1.course" && make -C "$build_dir" -j8
            time "$build_dir"/kaikai_datawhale_course1
            ;;
        2)
            echo "Running tests for course 2"
            build_dir="$SCRIPT_DIR/../build/2-course"
            echo "Build directory: $build_dir"
            cmake -B "$build_dir/" -S "$build_dir/../../2.course" && make -C "$build_dir" -j8
            time "$build_dir"/kaikai_datawhale_course2
            ;;

        3)
            echo "Running tests for course 3"
            ;;
        4)
            echo "Running tests for course 4"
            ;;

        5)
            echo "Running tests for course 5"
            ;;
        6)
            echo "Running tests for course 6"
            ;;

        7)
            echo "Running tests for course 7"
            ;;
        8)
            echo "Running tests for course 8"
            ;;

        9)
            echo "Running tests for course 9"
            ;;
        *)
            echo "Invalid project id: $project_id"
            exit 1
            ;;
    esac
}

# 检查传入的参数
if [[ $# -eq 0 ]]; then
    echo "No options provided."
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do # 遍历所有参数
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;   
        -c|--clean)
            echo "Cleaning the workspace"
            rm -rf "$SCRIPT_DIR/../build"
            shift
            ;;
        -t|--test)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Please specify the project id"
                show_help
                exit 1
            fi
            run_test "$1"
            echo 0
            ;;
        -a|--all)
            echo "Building all versions"
            ;;
        *)
            echo "Invalid option: "
            show_help
            exit 1
            ;;
    esac
    shift # 移除已处理的参数
done


