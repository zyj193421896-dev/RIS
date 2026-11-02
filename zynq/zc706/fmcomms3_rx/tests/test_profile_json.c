/* Exercise the actual firmware STATUS serializer without board hardware. */
#include "capture_store.h"
#include <stdio.h>
#include <string.h>

int main(void)
{
    char json[8192];
    int count;
    memset(&capture_store, 0, sizeof(capture_store));
    capture_store.state = CAP_STORE_EMPTY;
    count = capture_store_json(json, sizeof(json));
    if (count <= 0 || count >= (int)sizeof(json)) return 1;
    fputs(json, stdout);
    return 0;
}
