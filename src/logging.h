#ifndef __LOGGING_UTILS_H__
#define __LOGGING_UTILS_H__

#include <boost/log/trivial.hpp>
#define _TRACE BOOST_LOG_TRIVIAL(trace)
#define _DEBUG BOOST_LOG_TRIVIAL(debug)
#define _INFO  BOOST_LOG_TRIVIAL(info)
#define _WARN  BOOST_LOG_TRIVIAL(warning)
#define _ERROR BOOST_LOG_TRIVIAL(error)


void init_boost_log(bool verbose);

#endif  //  end for __LOGGING_UTILS_H__