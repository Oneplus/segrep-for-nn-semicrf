#include "logging.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>


void init_boost_log(bool verbose) {
  namespace logging = boost::log;
  namespace src = boost::log::sources;
  namespace expr = boost::log::expressions;
  namespace keywords = boost::log::keywords;

  logging::add_console_log(
    std::clog,
    keywords::format = (
    expr::stream
    << expr::format_date_time< boost::posix_time::ptime >(
    "TimeStamp",
    "%Y-%m-%d %H:%M:%S")
    << " [" << logging::trivial::severity << "] "
    << expr::smessage
    )
    );

  if (verbose) {
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::trace);
  } else {
    logging::core::get()->set_filter(logging::trivial::severity > logging::trivial::trace);
  }

  logging::add_common_attributes();
}


