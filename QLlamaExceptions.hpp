#ifndef QLLAMAEXCEPTIONS_HPP
#define QLLAMAEXCEPTIONS_HPP

#include <QString>

#include <stdexcept>
#include <exception>

class QFileIOException : public std::exception
{
public:
    QFileIOException(const QString &what) : _what(what.toLatin1()) {}

    const char *what() const throw() override { return _what; }


private:
    const char *_what;
};

class QInvalidHexException : public std::exception
{
public:
    QInvalidHexException(int size, const char *src) : size(size), src(src) {}

    const char *what() const throw() override
    {
        return ("Expecting " + QString::number(size) + " hex chars at " + QString::number(reinterpret_cast<quintptr>(src), 16)).toLatin1();
    }


private:
    int size;
    const char *src;
};

#endif // QLLAMAEXCEPTIONS_HPP
