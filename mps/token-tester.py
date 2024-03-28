from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline

mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

prompt = """/void _quicksort (void *const pbase, size_t total_elems, size_t size,
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
"""

print(f"Mistral Tokens Length: {len(mistral_tokenizer.encode(prompt))}")
print(f"Llama Tokens Length: {len(llama_tokenizer.encode(prompt))}")
print(f"Falcon Tokens Length: {len(falcon_tokenizer.encode(prompt))}")
