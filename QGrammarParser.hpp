#ifndef QGRAMMARPARSER_H
#define QGRAMMARPARSER_H

// Implements a Qt adaptation of the original Llama.cpp grammar parser.
// Original comment found in common/grammar_parser.h:
// Implements a parser for an extended Backus-Naur form (BNF), producing the
// binary context-free grammar format specified by llama.h. Supports character
// ranges, grouping, and repetition operators. As an example, a grammar for
// arithmetic might look like:
//
// root  ::= expr
// expr  ::= term ([-+*/] term)*
// term  ::= num | "(" space expr ")" space
// num   ::= [0-9]+ space
// space ::= [ \t\n]

#include <QLlamaDefines.h>
#include <QLlamaExceptions.hpp>

#include <llama.h>
#include <QList>
#include <QMap>
#include <QtTypes>
#include <QString>
#include <cwchar>
#include <utility>

namespace QLlamaGrammarParser {

    struct QParseState {
        QMap<QString, quint32>              symbolIds;
        QList<QList<llama_grammar_element>> rules;

        QList<const llama_grammar_element *> cRules();

        friend FILE *operator<<(FILE *file, QParseState const &state);
    };

    QParseState parse(const QString *src);
    void printGrammar(FILE *file, const QParseState &state);

    Q_USEMAYBE static QPair<QString, const char *> decodeUtf8(const char *src)
    {
        static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };

        quint8 firstByte = static_cast<quint8>(*src);
        quint8 highbits = firstByte >> 4;
        int len = lookup[highbits];
        quint8 mask = (1 << (8 - len)) - 1;
        quint32 value = firstByte & mask;
        const char *end = src + len; // WARNING: May overrun.
        const char * pos = src + 1;
        for (; pos < end && *pos; ++pos)
            value = (value << 6) + (static_cast<quint8>(*pos) & 0x3F);

        QByteArray byteValue = QByteArray::number(value);

        return QPair<QString, const char *>(QString::fromUtf8(byteValue), pos);
    }

    Q_USEMAYBE static quint32 getSymbolId(QParseState &state, const char *src, size_t len)
    {
        quint32 nextId = static_cast<quint32>(state.symbolIds.size());
        state.symbolIds.insert(QString::fromLatin1(src, len), nextId);
        return state.symbolIds.value(QString::fromLatin1(src, len));
    }

    Q_USEMAYBE static quint32 generateSymbolId(QParseState &state, const QString &baseName)
    {
        quint32 nextId = static_cast<quint32>(state.symbolIds.size());
        state.symbolIds[baseName + "_" + QString::number(nextId)] = nextId;

        return nextId;
    }

    Q_USEMAYBE static void addRule(
        QParseState &state,
        quint32 ruleId,
        const QList<llama_grammar_element> &rule
    )
    {
        if (state.rules.size() <= ruleId)
            state.rules.resize(ruleId + 1);
        state.rules[ruleId] = rule;
    }

    Q_USEMAYBE static bool isDigitChar(char c)
    {
        return '0' <= c && c <= '9';
    }

    Q_USEMAYBE static bool isWordChar(char c)
    {
        return    ('a' <= c && c <= 'z')
               || ('A' <= c && c <= 'Z')
               || c == '-'
               || isDigitChar(c);
    }

    Q_USEMAYBE static QPair<QString, const char *> parseHex(const char *src, int size)
    {
        const char *pos = src;
        const char *end = src + size;

        quint32 value = 0;

        for (; pos < end && *pos; ++pos)
        {
            value <<= 4;
            char c = *pos;

            if ('a' <= c && c <= 'f')
                value += c - 'a' + 10;
            else if ('A' <= c && c <= 'F')
                value += c - 'A' + 10;
            else if ('0' <= c && c <= '9')
                value += c - '0';
            else
                break;
        }

        if (pos != end)
            throw QInvalidHexException(size, src);

        QByteArray byteValue = QByteArray::number(value);

        return QPair<QString, const char *>(QString::fromUtf8(byteValue), pos);
    }
}

#endif // QGRAMMARPARSER_H
