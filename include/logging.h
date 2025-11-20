#ifndef __ED_LOGGING_H__
#define __ED_LOGGING_H__

#include <iostream>

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data);

#endif