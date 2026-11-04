using System.Runtime.InteropServices;
using System.Text;

namespace Markitai.Core;

/// <summary>The two core calls a conversion makes, so the queue can run against a fake.</summary>
public interface IConverter
{
    string ClaimOutputFolder(string request);
    string ConvertJson(string request);
}

/// <summary>markitai_app.dll (crates/markitai-app/include/markitai_app.h). Each call takes and answers JSON text.</summary>
public sealed unsafe partial class NativeCore : IConverter
{
    public const uint ExpectedAbiVersion = 2;
    const string Library = "markitai_app";

    public static NativeCore Instance { get; } = new();

    NativeCore() { }

    public static uint AbiVersion() => markitai_app_abi_version();

    public static string Version() => Marshal.PtrToStringUTF8(markitai_app_version()) ?? "";

    public static string PrepareEnvironment(string request) => Call(&markitai_app_prepare_environment, request);

    public static string PreviewHtml(string request) => Call(&markitai_app_preview_html, request);

    public static string MarkdownBody(string request) => Call(&markitai_app_markdown_body, request);

    public static string ConfigFile()
    {
        var buffer = markitai_app_config_file();
        return Take(&buffer);
    }

    public string ClaimOutputFolder(string request) => Call(&markitai_app_claim_output_folder, request);

    public string ConvertJson(string request) => Call(&markitai_app_convert_json, request);

    static string Call(delegate*<byte*, nuint, Buffer> function, string request)
    {
        var bytes = Encoding.UTF8.GetBytes(request);
        fixed (byte* data = bytes)
        {
            var buffer = function(data, (nuint)bytes.Length);
            return Take(&buffer);
        }
    }

    static string Take(Buffer* buffer)
    {
        try
        {
            return Encoding.UTF8.GetString(buffer->Data, checked((int)buffer->Length));
        }
        finally
        {
            markitai_app_buffer_free(buffer);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Buffer
    {
        public byte* Data;
        public nuint Length;
    }

    [LibraryImport(Library)]
    private static partial uint markitai_app_abi_version();

    [LibraryImport(Library)]
    private static partial nint markitai_app_version();

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_prepare_environment(byte* request, nuint length);

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_convert_json(byte* request, nuint length);

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_preview_html(byte* request, nuint length);

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_markdown_body(byte* request, nuint length);

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_claim_output_folder(byte* request, nuint length);

    [LibraryImport(Library)]
    private static partial Buffer markitai_app_config_file();

    [LibraryImport(Library)]
    private static partial void markitai_app_buffer_free(Buffer* buffer);
}
