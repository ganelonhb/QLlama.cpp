#ifndef QLLAMAINFERENCE_HPP
#define QLLAMAINFERENCE_HPP

#define Q_USEMAYBE [[maybe_unused]]
#define helpers

#include "common/common.h"
#include <llama.h>

#include <QObject>

#include <QList>
#include <QFile>
#include <QCoreApplication>
#include <QTextStream>

#include <string.h>
#include <exception>

namespace QLlamaExceptions
{
    class QCouldNotReadFileException : std::exception
    {
    public:
        QCouldNotReadFileException(const QString &err)
        {
            QString qWut = "Could not open file. Err: " + err;
            for (int i = 0; i < qWut.size(); ++i)
                wut[i] = qWut.at(i).toLatin1();
        };

        const char *what() const throw() override
        {

            return wut;
        }

    private:
        char wut[512] = {0};
    };

    class QLogfileError : std::exception
    {
    public:
        QLogfileError(const QString &func, const QString &logdir)
        {
            QString qWut = func + ": warning: failed to create logdir" + logdir + ", cannot write logfile\n";

            for (int i = 0; i < qWut.size(); ++i)
                wut[i] = qWut.at(i).toLatin1();
        }

        const char *what() const throw() override
        {
            return wut;
        }

    private:        
        char wut[512] = {0};
    };
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

            if (m_params.seed == LLAMA_DEFAULT_SEED)
                m_params.seed = time(NULL);

            m_sparams = m_params.sparams;

            std::mt19937 rng(m_params.seed);
        }
    }

    ~QLlamaInference()
    {
        if (m_ctx_guidance) llama_free(m_ctx_guidance);
        if (m_ctx) llama_free(m_ctx);
        if (m_model) llama_free_model(m_model);
        //if (m_ctx_sampling) llama_sampling_free(m_ctx_sampling);
        llama_backend_free();
    }

    llama_model *model()            { return m_model; }
    llama_context * ctx()           { return m_ctx; }
    gpt_params params()             { return m_params; }
    llama_sampling_params sparams() { return m_sparams; }
    int n_ctx_train()               { return m_n_ctx_train; }
    int n_ctx()                     { return m_n_ctx; }

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

    void set_n_threads(qint32 n_threads = cpu_get_num_math())           { m_params.n_threads = n_threads; }
    void set_n_threads_draft(qint32 n_threads_draft = -1)               { m_params.n_threads_draft = n_threads_draft; }
    void set_n_threads_batch(qint32 n_threads_batch = -1)               { m_params.n_threads_batch = n_threads_batch; }
    void set_n_threads_batch_draft(qint32 n_threads_batch_draft = -1)   { m_params.n_threads_batch_draft = n_threads_batch_draft; }
    void set_n_predict(qint32 n_predict = -1)                           { m_params.n_predict = n_predict; }
    void set_n_ctx(qint32 n_ctx = 0)                                    { m_params.n_ctx = n_ctx; }
    void set_n_batch(qint32 n_batch = 2048)                             { m_params.n_batch = n_batch; }
    void set_n_ubatch(qint32 n_ubatch = 512)                            { m_params.n_ubatch = n_ubatch; }
    void set_n_keep(qint32 n_keep = 0)                                  { m_params.n_keep = n_keep; }
    void set_n_draft(qint32 n_draft = 5)                                { m_params.n_draft = n_draft; }
    void set_n_chunks(qint32 n_chunks = -1)                             { m_params.n_chunks = n_chunks; }
    void set_n_parallel(qint32 n_parallel = 1)                          { m_params.n_parallel = n_parallel; }
    void set_n_sequences(qint32 n_sequences = 1)                        { m_params.n_sequences = n_sequences; }
    void set_p_split(float p_split = 0.1f)                              { m_params.p_split = p_split; }
    void set_n_gpu_layers(qint32 n_gpu_layers = -1)                     { m_params.n_gpu_layers = n_gpu_layers; }
    void set_n_gpu_layers_draft(qint32 n_gpu_layers_draft = -1)         { m_params.n_gpu_layers_draft = n_gpu_layers_draft; }
    void set_main_gpu(qint32 main_gpu = 0)                              { m_params.main_gpu = main_gpu; }
    void set_tensor_split(float tensor_split[128] = 0)                  { memset(&m_params.tensor_split, 0, sizeof(float) * 128); for(int i = 0; i < 128 || tensor_split[i] == '0'; ++i) m_params.tensor_split[i] = tensor_split[i]; }
    void set_grp_attn_n(qint32 grp_attn_n = 1)                          { m_params.grp_attn_n = grp_attn_n; }


private:
    gpt_params m_params;

    llama_model *m_model                    {nullptr};

    llama_context *m_ctx                    {nullptr};
    llama_sampling_params m_sparams;
    llama_sampling_context *ctx_sampling    {nullptr};
    llama_context *m_ctx_guidance           {nullptr};

    int m_n_ctx_train;
    int m_n_ctx;

    bool is_interacting                     {false};
    bool need_insert_eot                    {false};

    QList<llama_chat_msg> m_chat_msgs;

private helpers:

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

    void write_logfile(
        const QList<llama_token> &input_tokens,
        const QString &output,
        const QList<llama_token> &output_tokens
    )
    {
        if (!m_ctx || !m_model) return;

        if (m_params.logdir.empty()) return;

        const QString timestamp = QString::fromStdString(string_get_sortable_timestamp());

        const bool success = fs_create_directory_with_parents(m_params.logdir);

        if (!success) {
            fprintf(stderr, "%s: Warning: Failed to create logdir %s. Cannot write logfile.\n", __func__, m_params.logdir.c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), QString::fromLatin1(m_params.logdir));
        }

        const QString logfile_path = QString::fromLatin1(m_params.logdir) + timestamp + ".yml";

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
        llama_model_desc(m_model, model_desc, sizeof(model_desc));
        std::vector<llama_token> input_tokens_vec;
        input_tokens_vec.reserve(input_tokens.size());

        foreach(llama_token t, input_tokens)
            input_tokens_vec.push_back(t);

        int fileHandle = logfile.handle();

        FILE *fileDescriptor = fdopen(fileHandle, "a");

        if (!fileDescriptor)
        {
            fprintf(stderr, "%s: Warning: Failed to associate logfile %s. with file descriptor\n", __func__, logfile_path.toStdString().c_str());
            throw QLlamaExceptions::QLogfileError(QString::fromLatin1(__func__), QString::fromLatin1(m_params.logdir));
            logfile.close();
        }

        yaml_dump_non_result_info(fileDescriptor, m_params, m_ctx, timestamp.toStdString().c_str(), input_tokens_vec, model_desc);
        fflush(fileDescriptor);

        logStream << "\n"
                  << "######################\n"
                  << "# Generation Results #\n"
                  << "######################\n"
                  << "\n";
        logStream.flush();

        yaml_dump_string_multiline(fileDescriptor, "output", output.toStdString().c_str());
        fflush(fileDescriptor);

        std::vector<llama_token> output_tokens_vec;

        foreach(llama_token t, output_tokens)
            output_tokens_vec.push_back(t);

        yaml_dump_vector_int(fileDescriptor, "output_tokens", output_tokens_vec);
        fflush(fileDescriptor);

        llama_dump_timing_info_yaml(fileDescriptor, m_ctx);
        fflush(fileDescriptor);

        fclose(fileDescriptor);
        logfile.close();
    }

    void llama_log_callbck_logTee(ggml_log_level level, const QString &text, void *user_data)
    {
        Q_UNUSED(level);
        Q_UNUSED(user_data);

        LOG_TEE("%s", text.toStdString().c_str());
    }

    QString chat_add_and_format(QList<llama_chat_msg> &chat_msgs, QString role, QString content)
    {
        std::vector<llama_chat_msg> v_chat_msgs;
        foreach(llama_chat_msg m , chat_msgs)
            v_chat_msgs.push_back(m);

        std::string role_str = role.toStdString();
        std::string content_str = content.toStdString();

        llama_chat_msg new_msg{role_str, content_str};

        std::string formatted = llama_chat_format_single(
            m_model,
            m_params.chat_template,
            v_chat_msgs,
            new_msg,
            role == "user"
        );

        chat_msgs.push_back({role_str, content_str});
        LOG("Formatted: %s\n", formatted.c_str());

        return QString::fromStdString(formatted);
    }
};

#endif // QLLAMAINFERENCE_HPP
