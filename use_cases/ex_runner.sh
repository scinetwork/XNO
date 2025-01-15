#!/bin/bash

method="xno"
mix_mode=("pure")

transformations=("fno" "hno" "lno" "wno")
scenario_parallel=("fl" "fw" "wl" "fwl")
parallel_kernels_list=("fno lno" "fno wno" "wno lno" "fno wno lno")

scenario_pure=("f-l-w" "f-w-l" "l-f-w" "l-w-f" "w-f-l" "w-l-f")
pure_kernels_order_list=("fno lno wno" "fno wno lno" "lno fno wno" "lno wno fno" "wno fno lno" "wno lno fno")

# Other parameters
data_path="${HOME}/repos/XNO/use_cases_xyno/data/burgers_data_R10.mat"
dataset="1d_burgers"
save_out="true"

# Create a directory for logs if it doesn't exist
mkdir -p logs

# Counter for enumeration
counter=1

if [["$method" == "xno"]]; then
    for transformation in "${transformations[@]}"; do
        echo "Running experiment $counter with:"
        echo "Method: $method" 
        echo "Transformation: $transformation"

        echo "Executing: python ex_runner.py --data_path $data_path --dataset $dataset --method $method --transformation $transformation --save_out $save_out"

        # Run the Python script
        python ex_runner.py \
            --data_path $data_path \
            --dataset $dataset \
            --method $method \
            --transformation $transformation \
            --save_out $save_out \
            > /dev/null 2> "logs/${dataset}_${method}_${transformation}_error.log"

        # Check the exit status of the Python script
        if [ $? -ne 0 ]; then
            echo "Experiment $transformation failed. See logs/${dataset}_${method}_${transformation}_error.log for details."
        else
            echo "Experiment $transformation completed successfully."
        fi

        # Increment the counter
        counter=$((counter + 1))
    done

elif [["$method" == "xyno"]]; then
    # Iterate over all combinations of parameters
    for mode in "${mix_mode[@]}"; do
        if [[ "$mode" == "parallel" ]]; then
            for i in "${!scenario_parallel[@]}"; do
                current_scenario="${scenario_parallel[$i]}"
                current_parallel_kernels="${parallel_kernels_list[$i]}"

                echo "Running experiment $counter with:"
                echo "Method: $method"
                echo "Mix_mode: $mode"
                echo "Scenario: $current_scenario"
                echo "Parallel Kernels: $current_parallel_kernels"

                echo "Executing: python ex_runner.py --data_path $data_path --dataset $dataset --method $method --mix_mode $mode --scenario $current_scenario --parallel_kernels $current_parallel_kernels --save_out $save_out"

                # Run the Python script
                python ex_runner.py \
                    --data_path $data_path \
                    --dataset $dataset \
                    --method $method \
                    --mix_mode $mode \
                    --scenario $current_scenario \
                    --parallel_kernels $current_parallel_kernels \
                    --save_out $save_out \
                    > /dev/null 2> "logs/${dataset}_${method}_${mode}_${current_scenario}_error.log"

                # Check the exit status of the Python script
                if [ $? -ne 0 ]; then
                    echo "Experiment $current_scenario failed. See logs/${dataset}_${method}_${mode}_${current_scenario}_error.log for details."
                else
                    echo "Experiment $current_scenario completed successfully."
                fi

                # Increment the counter
                counter=$((counter + 1))
            done
        elif [[ "$mode" == "pure" ]]; then
            for i in "${!scenario_pure[@]}"; do
                current_scenario="${scenario_pure[$i]}"
                current_pure_kernels="${pure_kernels_order_list[$i]}"

                echo "Running experiment $counter with:"
                echo "Method: $method"
                echo "Mix_mode: $mode"
                echo "Scenario: $current_scenario"
                echo "Pure Kernels: $current_pure_kernels"

                echo "Executing: python ex_runner.py --data_path $data_path --dataset $dataset --method $method --mix_mode $mode --scenario $current_scenario --pure_kernels_order $current_pure_kernels --save_out $save_out"

                # Run the Python script
                python ex_runner.py \
                    --data_path $data_path \
                    --dataset $dataset \
                    --method $method \
                    --mix_mode $mode \
                    --scenario $current_scenario \
                    --pure_kernels_order $current_pure_kernels \
                    --save_out $save_out \
                    > /dev/null 2> "logs/${dataset}_${method}_${mode}_${current_scenario}_error.log"

                # Check the exit status of the Python script
                if [ $? -ne 0 ]; then
                    echo "Experiment $current_scenario failed. See logs/${dataset}_${method}_${mode}_${current_scenario}_error.log for details."
                else
                    echo "Experiment $current_scenario completed successfully."
                fi
                # Increment the counter
                counter=$((counter + 1))
            done
        else
            echo "The Mixing Mode of \"$mode\" is not recognized!"
        fi
    done

else
    echo "The Method of \"$method\" is not recognized!"
fi


# python ex_runner.py --data_path /home/hghadjari/repos/XNO/use_cases_xyno/data/burgers_data_R10.mat --dataset 1d_burgers --mix_mode parallel --scenario FL --parallel_kernels wno --pure_kernels_order wno --save_out false