#ifndef QLLAMAINFERENCE_HPP
#define QLLAMAINFERENCE_HPP

#define Q_USEMAYBE [[maybe_unused]]

#include "common/common.h"
#include <llama.h>

#include <QObject>

#include <QList>
#include <QFile>
#include <QCoreApplication>
#include <QTextStream>

#include <string.h>
#include <exception>

namespace QLlamaGlobals
{
    static llama_context            **g_ctx;
    static llama_context            **g_model;
    static gpt_params                *g_params;
    static QList<llama_token>        *g_input_tokens;
}

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
    Q_USEMAYBE static bool file_exists(const QString &path)
    {
        return QFile::exists(path);
    }

    Q_USEMAYBE static bool file_is_empty(const QString &path) noexcept(false)
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

    Q_USEMAYBE static void write_logfile(
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
            fprintf(stderr, "%s: Warning: Failed to create logdir %s. Cannot write logfile.\n", __func__, params.logdir.c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), QString::fromLatin1(params.logdir));
        }

        const QString logfile_path = QString::fromLatin1(params.logdir) + timestamp + ".yml";

        QFile logfile(logfile_path);

        if (!logfile.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            fprintf(stderr, "%s: Warning: Failed to open logfile %s.\n", __func__, logfile_path.toStdString().c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), logfile_path);
        }

        //logfile.write("Binary: " + QCoreApplication::applicationName());
        QTextStream logStream(&logfile);

        logStream << "Binary: " << QCoreApplication::applicationName() << '\n';
        logStream.flush();
        char model_desc[128];
        llama_model_desc(model, model_desc, sizeof(model_desc));
        std::vector<llama_token> input_tokens_vec;
        input_tokens_vec.reserve(input_tokens.size());

        foreach(llama_token t, input_tokens)
            input_tokens_vec.push_back(t);

        int fileHandle = logfile.handle();

        FILE *fileDescriptor = fdopen(fileHandle, "a");

        if (!fileDescriptor)
        {
            fprintf(stderr, "%s: Warning: Failed to associate logfile %s. with file descriptor\n", __func__, logfile_path.toStdString().c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), QString::fromLatin1(params.logdir));
            logfile.close();
        }

        yaml_dump_non_result_info(fileDescriptor, params, ctx, timestamp.toStdString().c_str(), input_tokens_vec, model_desc);
        fflush(fileDescriptor);

        logStream << "\n"
                  << "######################\n"
                  << "# Generation Results #\n"
                  << "######################\n"
                  << "\n";
        logStream.flush();

        yaml_dump_string_multiline(fileDescriptor, "output", output.toStdString().c_str());
        fflush(fileDescriptor);

        yaml_dump_vector_int(fileDescriptor, "output_tokens", input_tokens_vec);
        fflush(fileDescriptor);

        llama_dump_timing_info_yaml(fileDescriptor, ctx);
        fflush(fileDescriptor);

        fclose(fileDescriptor);
        logfile.close();
    }

    Q_USEMAYBE static void llama_log_callbck_logTee(ggml_log_level level, const QString &text, void *user_data)
    {
        Q_UNUSED(level);
        Q_UNUSED(user_data);

        LOG_TEE("%s", text.toStdString().c_str());
    }

    Q_USEMAYBE static QString chat_add_and_format(struct llama_model *model, QList<llama_chat_msg> &chat_msgs, QString role, QString content)
    {
        std::vector<llama_chat_msg> v_chat_msgs;
        foreach(llama_chat_msg m , chat_msgs)
            v_chat_msgs.push_back(m);

        std::string role_str = role.toStdString();
        std::string content_str = content.toStdString();

        llama_chat_msg new_msg{role_str, content_str};

        std::string formatted = llama_chat_format_single(
            model,
            QLlamaGlobals::g_params->chat_template,
            v_chat_msgs,
            new_msg,
            role == "user"
        );

        chat_msgs.push_back({role_str, content_str});
        LOG("Formatted: %s\n", formatted.c_str());

        return QString::fromStdString(formatted);
    }
}

class QLlamaInference : QObject
{
    Q_OBJECT

public:
    QLlamaInference(gpt_params *p = nullptr, QObject *parent = nullptr)
        : QObject(parent)
    {
        if (p)
        {
            m_params = *p;
            QLlamaGlobals::g_params = p;

            if (m_params.seed == LLAMA_DEFAULT_SEED)
                m_params.seed = time(NULL);

            m_sparams = m_params.sparams;

            std::mt19937 rng(m_params.seed);
        }
    }

    llama_model *model() { return m_model; }
    llama_context * ctx() { return m_ctx; }
    gpt_params params() { return m_params; }
    llama_sampling_params sparams() { return m_sparams; }
    int n_ctx_train() { return m_n_ctx_train; }
    int n_ctx() { return m_n_ctx; }

    QList<llama_chat_msg> &chat_msgs() { return m_chat_msgs; }

    // Setters (lots of them)
    void setParams(gpt_params params)
    {
        m_params = params;

        if (m_params.seed == LLAMA_DEFAULT_SEED)
            m_params.seed = time(NULL);

        m_sparams = m_params.sparams;

        std::mt19937 rng(m_params.seed);
    }

    void setSeed(quint32 seed = LLAMA_DEFAULT_SEED) {
        m_params.seed = (seed == LLAMA_DEFAULT_SEED) ? time(NULL) : seed;
        std::mt19937 rng(m_params.seed);
    }
    void setN_threads(qint32 n_threads = cpu_get_num_math())            { m_params.n_threads = n_threads; }
    void setN_threads_draft(qint32 n_threads_draft = -1)                { m_params.n_threads_draft = n_threads_draft; }
    void setN_threads_batch(qint32 n_threads_batch = -1)                { m_params.n_threads_batch = n_threads_batch; }
    void setN_threads_batch_draft(qint32 n_threads_batch_draft = -1)    { m_params.n_threads_batch_draft = n_threads_batch_draft; }
    void setN_predict(qint32 n_predict = -1)                            { m_params.n_predict = n_predict; }
    void setN_ctx(qint32 n_ctx = 0)                                     { m_params.n_ctx = n_ctx; }
    void setN_batch(qint32 n_batch = 2048)                              { m_params.n_batch = n_batch; }
    void setN_ubatch(qint32 n_ubatch = 512)                             { m_params.n_ubatch = n_ubatch; }
    void setN_keep(qint32 n_keep = 0)                                   { m_params.n_keep = n_keep; }
    void setN_draft(qint32 n_draft = 5)                                 { m_params.n_draft = n_draft; }
    void setN_chunks(qint32 n_chunks = -1)                              { m_params.n_chunks = n_chunks; }
    void setN_parallel(qint32 n_parallel = 1)                           { m_params.n_parallel = n_parallel; }

private:
    gpt_params m_params;

    llama_model *m_model;

    llama_context *m_ctx;
    llama_sampling_params m_sparams;
    llama_context *m_ctx_guidance {nullptr};

    int m_n_ctx_train;
    int m_n_ctx;

    QList<llama_chat_msg> m_chat_msgs;
};

#endif // QLLAMAINFERENCE_HPP
