import Foundation

/// A conversion's folder as its preview reads it, at a scheme the app serves.
public enum ResultFolder {
    public static let scheme = "markitai-preview"
    public static let url = "\(scheme)://result/"

    /// The file a URL's path names inside `folder`, links followed. Nothing
    /// for a folder, a missing file or anything outside it.
    public static func file(path: String, in folder: URL) -> URL? {
        let root = folder.resolvingSymlinksInPath()
        let file = root.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        var isFolder: ObjCBool = false
        guard file.path.hasPrefix(root.path + "/"),
              FileManager.default.fileExists(atPath: file.path, isDirectory: &isFolder),
              !isFolder.boolValue
        else { return nil }
        return file
    }
}
