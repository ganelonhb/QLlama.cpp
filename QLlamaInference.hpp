#ifndef QLLAMAINFERENCE_HPP
#define QLLAMAINFERENCE_HPP

#define Q_USEMAYBE [[maybe_unused]]

#include "common/common.h"
#include <llama.h>

#include <QObject>

#include <QList>
#include <QFile>

#include <string.h>
#include <exception>

namespace QLlamaExceptions
{
    class QCouldNotReadFileException : std::exception
    {
    public:
        QCouldNotReadFileException(const QString &err)
            : err(err) {};

        const char *what() const throw() override
        {
            return ("Could not open file. Err: " + err).toLatin1();
        }

    private:
        const QString err;
    };

    class QLogfileError : std::exception
    {
    public:
        QLogfileError(const QString &func, const QString &logdir)
            : func(func)
            , logdir(logdir)
        {}

        const char *what() const throw() override
        {
            return (func + ": warning: failed to create logdir" + logdir + ", cannot write logfile\n").toLatin1();
        }

    private:
        const QString& func;
        const QString& logdir;
    };
}

namespace QLlamaInferenceHelpers
{
    static bool file_exists(const QString &path)
    {
        return QFile::exists(path);
    }

    static bool file_is_empty(const QString &path) noexcept(false)
    {
        if (!file_exists(path))
            throw QLlamaExceptions::QCouldNotReadFileException("Could not open file. Err: The file does not exist.");

        QFile f(path);

        if (!f.open(QIODevice::ReadOnly))
            throw QLlamaExceptions::QCouldNotReadFileException(f.errorString());

        bool empty = !f.size();

        f.close();

        return empty;
    }

    static void write_logfile(
        const llama_context *ctx,
        const gpt_params &params,
        const llama_model *model,
        const QList<llama_token> &input_tokens,
        const QString &output,
        const QList<llama_token> &output_tokens
        )
    {
        if (params.logdir.empty()) return;

        const QString timestamp = QString::fromStdString(string_get_sortable_timestamp());

        const bool success = fs_create_directory_with_parents(params.logdir);

        if (!success) {
            fprintf(stderr, "%s: Warning: failed to create logdir %s. Cannot write logfile.\n", __func__, params.logdir.c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), QString::fromLatin1(params.logdir));
        }
    }
}

class QLlamaInference : QObject
{
    Q_OBJECT

public:
    QLlamaInference(const gpt_params *p = nullptr, QObject *parent = nullptr)
        : QObject(parent)
    {
        if (p)
        {
            m_params = *p;

            if (m_params.seed == LLAMA_DEFAULT_SEED)
                m_params.seed = time(NULL);

            m_sparams = m_params.sparams;

            std::mt19937 rng(m_params.seed);

            llama_backend_init();
            llama_numa_init(m_params.numa);

            std::tie(m_model, m_ctx) = llama_init_from_gpt_params(m_params);

            if (m_sparams.cfg_scale > 1.f)
            {
                struct llama_context_params lparams = llama_context_params_from_gpt_params(m_params);
                m_ctx_guidance = llama_new_context_with_model(m_model, lparams);
            }

            n_ctx_train = llama_n_ctx_train(m_model);
            n_ctx = llama_n_ctx(m_ctx);

            QString path_session = QString::fromStdString(m_params.path_prompt_cache);
            QList<llama_token> session_tokens;
        }
    }

    llama_model *model() { return m_model; }
    llama_context * ctx() { return m_ctx; }

private:
    gpt_params m_params;

    llama_model *m_model;
    llama_context *m_ctx;
    llama_sampling_params m_sparams;
    llama_context *m_ctx_guidance {nullptr};

    int n_ctx_train;
    int n_ctx;

    QList<llama_chat_msg> m_chat_msgs;
};

#endif // QLLAMAINFERENCE_HPP
