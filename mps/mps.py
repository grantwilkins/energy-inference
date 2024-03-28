from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
import torch
import subprocess
import threading
import re
import pandas as pd
from time import sleep
import time
import argparse
import os
import datetime
import torch.mps
import numpy as np
from scipy import stats


def load_model(
    model_name: str,
    load_model_event: threading.Event,
    load_model_thread: threading.Thread,
) -> tuple[Pipeline, AutoTokenizer, float]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_model_thread.start()
    load_model_start_time = time.time()
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    load_model_time = time.time() - load_model_start_time
    load_model_event.set()
    load_model_thread.join()

    return (
        pipe,
        tokenizer,
        load_model_time,
    )


def run_inference(
    pipe: Pipeline,
    num_tokens: int,
    prompt: str,
    batch_size: int,
    inference_event: threading.Event,
    inference_monitor: threading.Thread,
) -> str:
    inference_monitor.start()
    sleep(2)
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=num_tokens,
        min_new_tokens=int(num_tokens * 0.9),
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        use_cache=False,
        batch_size=batch_size,
    )
    inference_event.set()
    inference_monitor.join()

    return sequences[0]["generated_text"]


def monitor_power_usage(power_readings: list[str], stop_monitoring: threading.Event):
    cmd = "echo ***REMOVED*** | sudo -S powermetrics --show-process-energy -i 200"
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True, text=True
    ) as process:
        while not stop_monitoring.is_set():
            line = process.stdout.readline()
            if not line:
                break
            power_readings.append(line)
    print("Powermetrics process ended.")


def process_data(power_readings, event):
    cpu_pattern = r"CPU Power:\s+(\d+)\s+mW"
    gpu_pattern = r"GPU Power:\s+(\d+)\s+mW"
    keywords = ["python3.12", "ALL_TASKS"]
    readings = []
    power_usage = {"event": event}
    for line in power_readings:
        if not line:
            break
        cpu_match = re.search(cpu_pattern, line)
        gpu_match = re.search(gpu_pattern, line)
        if any(line.startswith(keyword) for keyword in keywords):
            split_line = line.split()
            power_usage[split_line[0]] = split_line[-1]
        if line.startswith("*** Sampled"):
            power_usage["timestamp"] = float(
                line.split()[-3].replace("(", "").replace("ms", "")
            )
        if cpu_match:
            power_usage["cpu"] = int(cpu_match.group(1))
        if gpu_match:
            power_usage["gpu"] = int(gpu_match.group(1))
        if power_usage.keys() == {
            "cpu",
            "gpu",
            "python3.12",
            "ALL_TASKS",
            "event",
            "timestamp",
        }:
            readings.append(power_usage)
            power_usage = {"event": event}
    return readings


def post_process_power_data(readings: list[dict], runtime_s: float) -> pd.DataFrame:
    df = pd.DataFrame()
    df["Event"] = [readings[0]["event"]]
    df["Runtime (s)"] = [runtime_s]
    total_cpu_energy = 0
    for reading in readings:
        total_cpu_energy += (
            reading["cpu"]
            * reading["timestamp"]
            * (float(reading["python3.12"]) / float(reading["ALL_TASKS"]))
            / 1000
            / 1000
        )
    df["CPU Energy (J)"] = [total_cpu_energy]
    total_gpu_energy = 0
    for reading in readings:
        total_gpu_energy += reading["gpu"] * reading["timestamp"] / 1000 / 1000
    df["GPU Energy (J)"] = [total_gpu_energy]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--hf_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--system_name", type=str, default="M1-Pro")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    todays_date = datetime.date.today().strftime("%Y-%m-%d")
    start_time = datetime.datetime.now().strftime("%H-%M-%S")
    num_tokens = args.num_tokens
    hf_name = args.hf_name
    system_name = args.system_name
    batch_size = args.batch_size
    model_name = args.hf_name.split("/")[1]
    prompts = {
        "A": "What is the largest city in France?",
        "B": "Can you explain the difference between a simile and a metaphor? Provide an example.",
        "C": "What are some effective strategies for managing stress and maintaining good mental health during challenging times, such as a pandemic, a break-up, or a personal crisis?",
        "D": """Imagine you are an expert travel guide for Japan. 
            Can you recommend a 7-day itinerary for a trip to Japan, including must-visit destinations, cultural experiences, and local tasty cuisine? 
            Provide a brief description of each day's activities and how they showcase the best of Japan.
            """,
        "E": """As an AI language model, you possess the capability to process and generate text that mimics human communication. 
        This unique ability allows you to explore the potential implications of advanced AI systems across a wide range of industries, including but not limited to healthcare, education, and various creative fields.
        In discussing these implications, it's crucial to consider the multifaceted benefits such as increased efficiency, personalized experiences, and the democratization of knowledge. 
        To provide a comprehensive analysis, you will delve into specific examples that highlight both the impacts and the challenges posed by the integration of AI technologies in these critical areas of society.",
        """,
        "F": """/* An improved random number generation package.  In addition to the standard
rand()/srand() like interface, this package also has a special state info
interface.  The initstate() routine is called with a seed, an array of
bytes, and a count of how many bytes are being passed in; this array is
then initialized to contain information for random number generation with
that much state information.  Good sizes for the amount of state
information are 32, 64, 128, and 256 bytes.  The state can be switched by
calling the setstate() function with the same array as was initialized
with initstate().  By default, the package runs with 128 bytes of state
information and generates far better random numbers than a linear
congruential generator.  If the amount of state information is less than
32 bytes, a simple linear congruential R.N.G. is used.  Internally, the
state information is treated as an array of longs; the zeroth element of
the array is the type of R.N.G.. */

Please describe what this random number generator library does in a few sentences. Please also provide some coding examples.
""",
        "G": """/*This is a version (aka ptmalloc2) of malloc/free/realloc written by
Doug Lea and adapted to multiple threads/arenas by Wolfram Gloger.

There have been substantial changes made after the integration into
glibc in all parts of the code.  Do not look for much commonality
with the ptmalloc2 version.

* Quickstart

In order to compile this implementation, a Makefile is provided with
the ptmalloc2 distribution, which has pre-defined targets for some
popular systems (e.g. "make posix" for Posix threads).  All that is
typically required with regard to compiler flags is the selection of
the thread package via defining one out of USE_PTHREADS, USE_THR or
USE_SPROC.  Check the thread-m.h file for what effects this has.
Many/most systems will additionally require USE_TSD_DATA_HACK to be
defined, so this is the default for "make posix".

* Why use this malloc?

This is not the fastest, most space-conserving, most portable, or
most tunable malloc ever written. However it is among the fastest
while also being among the most space-conserving, portable and tunable.
Consistent balance across these factors results in a good general-purpose
allocator for malloc-intensive programs.

The main properties of the algorithms are:
* For large (>= 512 bytes) requests, it is a pure best-fit allocator,
    with ties normally decided via FIFO (i.e. least recently used).
* For small (<= 64 bytes by default) requests, it is a caching
    allocator, that maintains pools of quickly recycled chunks.
* In between, and for combinations of large and small requests, it does
    the best it can trying to meet both goals at once.
* For very large requests (>= 128KB by default), it relies on system
    memory mapping facilities, if supported.

For a longer but slightly out of date high-level description, see
    http://gee.cs.oswego.edu/dl/html/malloc.html


This is a library header for some C library code, describe what it does and how to implement it.
Provide coding examples as well.
            """,
        "H": """/void _quicksort (void *const pbase, size_t total_elems, size_t size,
    __compar_d_fn_t cmp, void *arg)
{
char *base_ptr = (char *) pbase;

const size_t max_thresh = MAX_THRESH * size;

if (total_elems == 0)
/* Avoid lossage with unsigned arithmetic below.  */
return;

if (total_elems > MAX_THRESH)
{
char *lo = base_ptr;
char *hi = &lo[size * (total_elems - 1)];
stack_node stack[STACK_SIZE];
stack_node *top = stack;

PUSH (NULL, NULL);

while (STACK_NOT_EMPTY)
    {
    char *left_ptr;
    char *right_ptr;

/* Select median value from among LO, MID, and HI. Rearrange
    LO and HI so the three values are sorted. This lowers the
    probability of picking a pathological pivot value and
    skips a comparison for both the LEFT_PTR and RIGHT_PTR in
    the while loops. */

char *mid = lo + size * ((hi - lo) / size >> 1);

if ((*cmp) ((void *) mid, (void *) lo, arg) < 0)
    SWAP (mid, lo, size);
if ((*cmp) ((void *) hi, (void *) mid, arg) < 0)
    SWAP (mid, hi, size);
else
    goto jump_over;
if ((*cmp) ((void *) mid, (void *) lo, arg) < 0)
    SWAP (mid, lo, size);
jump_over:;

left_ptr  = lo + size;
right_ptr = hi - size;

/* Here's the famous ``collapse the walls'' section of quicksort.
    Gotta like those tight inner loops!  They are the main reason
    that this algorithm runs much faster than others. */
do
    {
    while ((*cmp) ((void *) left_ptr, (void *) mid, arg) < 0)
    left_ptr += size;

    while ((*cmp) ((void *) mid, (void *) right_ptr, arg) < 0)
    right_ptr -= size;

    if (left_ptr < right_ptr)
    {
    SWAP (left_ptr, right_ptr, size);
    if (mid == left_ptr)
        mid = right_ptr;
    else if (mid == right_ptr)
        mid = left_ptr;
    left_ptr += size;
    right_ptr -= size;
    }
    else if (left_ptr == right_ptr)
    {
    left_ptr += size;
    right_ptr -= size;
    break;
    }
    }
while (left_ptr <= right_ptr);

    /* Set up pointers for next iteration.  First determine whether
        left and right partitions are below the threshold size.  If so,
        ignore one or both.  Otherwise, push the larger partition's
        bounds on the stack and continue sorting the smaller one. */

    if ((size_t) (right_ptr - lo) <= max_thresh)
        {
        if ((size_t) (hi - left_ptr) <= max_thresh)
    /* Ignore both small partitions. */
            POP (lo, hi);
        else
    /* Ignore small left partition. */
            lo = left_ptr;
        }
    else if ((size_t) (hi - left_ptr) <= max_thresh)
        hi = right_ptr;
    else if ((right_ptr - lo) > (hi - left_ptr))
        {
        PUSH (lo, right_ptr);
        lo = left_ptr;
        }
    else
        {
        PUSH (left_ptr, hi);
        hi = right_ptr;
        }
    }
}
Please describe what the following quicksort algorithm and code does and how it works with some examples please.
""",
        "I": """#include <alloca.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Byte-wise swap two items of size SIZE. */
#define SWAP(a, b, size)						      \
  do									      \
    {									      \
      size_t __size = (size);						      \
      char *__a = (a), *__b = (b);					      \
      do								      \
	{								      \
	  char __tmp = *__a;						      \
	  *__a++ = *__b;						      \
	  *__b++ = __tmp;						      \
	} while (--__size > 0);						      \
    } while (0)

#define MAX_THRESH 4

/* Stack node declarations used to store unfulfilled partition obligations. */
typedef struct
  {
    char *lo;
    char *hi;
  } stack_node;

#define STACK_SIZE	(CHAR_BIT * sizeof(size_t))
#define PUSH(low, high)	((void) ((top->lo = (low)), (top->hi = (high)), ++top))
#define	POP(low, high)	((void) (--top, (low = top->lo), (high = top->hi)))
#define	STACK_NOT_EMPTY	(stack < top)


/* Order size using quicksort.  This implementation incorporates
   four optimizations discussed in Sedgewick:

   1. Non-recursive, using an explicit stack of pointer that store the
      next array partition to sort.  To save time, this maximum amount
      of space required to store an array of SIZE_MAX is allocated on the
      stack.  Assuming a 32-bit (64 bit) integer for size_t, this needs
      only 32 * sizeof(stack_node) == 256 bytes (for 64 bit: 1024 bytes).
      Pretty cheap, actually.

   2. Chose the pivot element using a median-of-three decision tree.
      This reduces the probability of selecting a bad pivot value and
      eliminates certain extraneous comparisons.

   3. Only quicksorts TOTAL_ELEMS / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.

   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (total_elems)
      stack size is needed (actually O(1) in this case)!  */

void
_quicksort (void *const pbase, size_t total_elems, size_t size,
	    __compar_d_fn_t cmp, void *arg)
{
  char *base_ptr = (char *) pbase;

  const size_t max_thresh = MAX_THRESH * size;

  if (total_elems == 0)
    /* Avoid lossage with unsigned arithmetic below.  */
    return;

  if (total_elems > MAX_THRESH)
    {
      char *lo = base_ptr;
      char *hi = &lo[size * (total_elems - 1)];
      stack_node stack[STACK_SIZE];
      stack_node *top = stack;

      PUSH (NULL, NULL);

      while (STACK_NOT_EMPTY)
        {
          char *left_ptr;
          char *right_ptr;

	  /* Select median value from among LO, MID, and HI. Rearrange
	     LO and HI so the three values are sorted. */

	  char *mid = lo + size * ((hi - lo) / size >> 1);

	  if ((*cmp) ((void *) mid, (void *) lo, arg) < 0)
	    SWAP (mid, lo, size);
	  if ((*cmp) ((void *) hi, (void *) mid, arg) < 0)
	    SWAP (mid, hi, size);
	  else
	    goto jump_over;
	  if ((*cmp) ((void *) mid, (void *) lo, arg) < 0)
	    SWAP (mid, lo, size);
	jump_over:;

	  left_ptr  = lo + size;
	  right_ptr = hi - size;

	  /* Here's the famous ``collapse the walls'' section of quicksort.
	     Gotta like those tight inner loops!  They are the main reason
	     that this algorithm runs much faster than others. */
	  do
	    {
	      while ((*cmp) ((void *) left_ptr, (void *) mid, arg) < 0)
		left_ptr += size;

	      while ((*cmp) ((void *) mid, (void *) right_ptr, arg) < 0)
		right_ptr -= size;

	      if (left_ptr < right_ptr)
		{
		  SWAP (left_ptr, right_ptr, size);
		  if (mid == left_ptr)
		    mid = right_ptr;
		  else if (mid == right_ptr)
		    mid = left_ptr;
		  left_ptr += size;
		  right_ptr -= size;
		}
	      else if (left_ptr == right_ptr)
		{
		  left_ptr += size;
		  right_ptr -= size;
		  break;
		}
	    }
	  while (left_ptr <= right_ptr);

          if ((size_t) (right_ptr - lo) <= max_thresh)
            {
              if ((size_t) (hi - left_ptr) <= max_thresh)
		/* Ignore both small partitions. */
                POP (lo, hi);
              else
		/* Ignore small left partition. */
                lo = left_ptr;
            }
          else if ((size_t) (hi - left_ptr) <= max_thresh)
	    /* Ignore small right partition. */
            hi = right_ptr;
          else if ((right_ptr - lo) > (hi - left_ptr))
            {
	      /* Push larger left partition indices. */
              PUSH (lo, right_ptr);
              lo = left_ptr;
            }
          else
            {
	      /* Push larger right partition indices. */
              PUSH (left_ptr, hi);
              hi = right_ptr;
            }
        }
    }


#define min(x, y) ((x) < (y) ? (x) : (y))

  {
    char *const end_ptr = &base_ptr[size * (total_elems - 1)];
    char *tmp_ptr = base_ptr;
    char *thresh = min(end_ptr, base_ptr + max_thresh);
    char *run_ptr;


    for (run_ptr = tmp_ptr + size; run_ptr <= thresh; run_ptr += size)
      if ((*cmp) ((void *) run_ptr, (void *) tmp_ptr, arg) < 0)
        tmp_ptr = run_ptr;

    if (tmp_ptr != base_ptr)
      SWAP (tmp_ptr, base_ptr, size);

    run_ptr = base_ptr + size;
    while ((run_ptr += size) <= end_ptr)
      {
	tmp_ptr = run_ptr - size;
	while ((*cmp) ((void *) run_ptr, (void *) tmp_ptr, arg) < 0)
	  tmp_ptr -= size;

	tmp_ptr += size;
        if (tmp_ptr != run_ptr)
          {
            char *trav;

	    trav = run_ptr + size;
	    while (--trav >= run_ptr)
              {
                char c = *trav;
                char *hi, *lo;

                for (hi = lo = trav; (lo -= size) >= tmp_ptr; hi = lo)
                  *hi = *lo;
                *hi = c;
              }
          }
      }
  }
}
Describe the code above and some potential confusion points for developers. Describe ways that we can also make the code more readable.
""",
    }

    out_dir = f"{model_name}/{todays_date}/{start_time}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_power = f"{out_dir}/{model_name}-{system_name}-power.csv"

    with open(f"{out_dir}/job_info.yaml", "w") as file:
        file.write("job:\n")
        file.write(f"  date: {todays_date}\n")
        file.write(f"  start_time: {start_time}\n")
        file.write("  details:\n")
        file.write(f"    model_name: {model_name}\n")
        file.write(f"    system_name: {system_name}\n")
        file.write(f"    num_tokens: {num_tokens}\n")
        file.write(f"    batch_size: {batch_size}\n")
        file.write(f"    hf_name: {hf_name}\n")
        for idx, prompt in prompts.items():
            file.write(f"    prompt-{idx}: {prompt[:50].strip()}\n")

    load_model_event = threading.Event()
    power_readings = {
        "load model": [],
    }

    load_model_monitor_thread = threading.Thread(
        target=monitor_power_usage,
        args=(power_readings["load model"], load_model_event),
    )

    pre_mem = torch.mps.current_allocated_memory() / 1024**2
    pipe, tokenizer, model_load_runtime = load_model(
        model_name=args.hf_name,
        load_model_event=load_model_event,
        load_model_thread=load_model_monitor_thread,
    )
    model_mem = torch.mps.current_allocated_memory() / 1024**2  # in MB
    readings_load_model = process_data(
        power_readings=power_readings["load model"], event="load model"
    )
    df_energy = post_process_power_data(
        readings=readings_load_model, runtime_s=model_load_runtime
    )
    df_energy["Output Token Limit"] = num_tokens
    df_energy["Input Tokens"] = 0
    df_energy["Iteration"] = 0
    df_energy["Model Name"] = model_name
    df_energy["Number of GPUs"] = 1
    df_energy["Prompt"] = ""
    df_energy["Output Tokens"] = 0
    df_energy["Batch Size"] = batch_size
    df_energy["System"] = system_name
    df_energy["GPU Memory (MB)"] = [model_mem - pre_mem]
    df_energy.to_csv(
        f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
    )

    df_power = pd.DataFrame(readings_load_model)
    # print(df_power)
    df_power.to_csv(
        csv_power,
        index=False,
        header=False,
        mode="a",
    )

    for idx, prompt in prompts.items():
        runtimes = []
        for i in range(100):
            dict_key = f"inference-{idx}-{i}"
            power_readings[dict_key] = []
            inference_event = threading.Event()
            inference_monitor = threading.Thread(
                target=monitor_power_usage,
                args=(power_readings[dict_key], inference_event),
            )
            inference_start_time = time.time()
            llm_output = run_inference(
                pipe=pipe,
                num_tokens=num_tokens,
                prompt=prompt,
                inference_event=inference_event,
                inference_monitor=inference_monitor,
                batch_size=batch_size,
            )
            inference_runtime = time.time() - inference_start_time
            inference_mem = torch.mps.current_allocated_memory() / 1024**2
            readings = process_data(power_readings[dict_key], dict_key)
            df_power = pd.DataFrame(readings)
            df_power.to_csv(
                csv_power,
                index=False,
                header=False,
                mode="a",
            )
            torch.mps.empty_cache()

            input_tokens = tokenizer.encode(prompt)
            num_input_tokens = len(input_tokens)
            output_tokens = tokenizer.encode(llm_output)
            num_output_tokens = len(output_tokens)
            df_energy_inference = post_process_power_data(readings, inference_runtime)
            df_energy_inference["Output Token Limit"] = num_tokens
            df_energy_inference["Input Tokens"] = num_input_tokens
            df_energy_inference["Iteration"] = i
            df_energy_inference["Model Name"] = model_name
            df_energy_inference["Number of GPUs"] = 1
            df_energy_inference["Prompt"] = prompt[:50].strip()
            df_energy_inference["Output Tokens"] = num_output_tokens
            df_energy_inference["Batch Size"] = batch_size
            df_energy_inference["System"] = system_name
            df_energy_inference["GPU Memory (MB)"] = inference_mem
            df_energy_inference.to_csv(
                f"{model_name}-{system_name}.csv", index=False, header=False, mode="a"
            )
            if i > 0:
                runtimes.append(inference_runtime)
            mean_runtime = np.mean(runtimes)
            std_err = stats.sem(runtimes)
            z_critical = stats.norm.ppf((1 + 0.95) / 2)
            ci_half_width = z_critical * std_err
            # Break if we have more than 5 samples and the confidence interval half-width is less than 0.5
            if i > 5 and ci_half_width < 0.5:
                break
