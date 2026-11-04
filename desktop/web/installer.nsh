; Markitai Web's additions to the Windows installer that mygo build writes
; (windows.installerInclude of mygo.json). mygo includes this file after
; MUI2.nsh, the pages and the languages, runs customInit in .onInit and
; customInstall at the end of the install.
;
; - A folder that holds files, and is not Markitai Web's, is never the
;   install folder: the uninstaller removes the install folder with all it
;   holds (RMDir /r of mygo's script), so the user's files would go with it.
;   The directory page refuses it, and a silent install (/S /D=...) fails.
; - A running Markitai Web is asked through its single-instance lock
;   (--is-running), told what quitting it means, and quit at once
;   (--quit-now): mygo's own close, CloseMainWindow and then Stop-Process
;   after 10 s, would show the app's question about unfinished conversions,
;   whose answer the forced stop then ignores.

!define MARKITAI_WEB_EXE "Markitai Web.exe"
!define MARKITAI_WEB_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\dev.markitai.web"

LangString MARKITAI_WEB_FOREIGN ${LANG_ENGLISH} "This folder holds other files. Choose an empty folder, or the one Markitai Web is installed in: uninstalling removes the install folder with everything in it."
LangString MARKITAI_WEB_RUNNING ${LANG_ENGLISH} "Markitai Web is running. Setup will quit it, and conversions it has not finished will be stopped."
LangString MARKITAI_WEB_RUNNING_UN ${LANG_ENGLISH} "Markitai Web is running. Uninstalling will quit it, and conversions it has not finished will be stopped."
!ifdef LANG_SIMPCHINESE
LangString MARKITAI_WEB_FOREIGN ${LANG_SIMPCHINESE} "这个文件夹里有其他文件。请选择一个空文件夹，或 Markitai Web 已安装的文件夹：卸载时会删除安装文件夹及其中的所有内容。"
LangString MARKITAI_WEB_RUNNING ${LANG_SIMPCHINESE} "Markitai Web 正在运行。安装程序会退出它，尚未完成的转换会被停止。"
LangString MARKITAI_WEB_RUNNING_UN ${LANG_SIMPCHINESE} "Markitai Web 正在运行。卸载会退出它，尚未完成的转换会被停止。"
!endif
!ifdef LANG_JAPANESE
LangString MARKITAI_WEB_FOREIGN ${LANG_JAPANESE} "このフォルダーには他のファイルがあります。空のフォルダーか、Markitai Web がインストールされているフォルダーを選んでください。アンインストールすると、インストール先のフォルダーは中身ごと削除されます。"
LangString MARKITAI_WEB_RUNNING ${LANG_JAPANESE} "Markitai Web が実行中です。セットアップは Markitai Web を終了し、完了していない変換は停止されます。"
LangString MARKITAI_WEB_RUNNING_UN ${LANG_JAPANESE} "Markitai Web が実行中です。アンインストールすると Markitai Web は終了し、完了していない変換は停止されます。"
!endif

Var MarkitaiWebIntro

; Leaves 1 on the stack when $INSTDIR holds files and is not Markitai Web's
; folder, else 0. Markitai Web's is the one its uninstall entry names, or
; one holding both its program and its uninstaller.
!macro MarkitaiWebForeign un
Function ${un}MarkitaiWebForeign
  Push $0
  Push $1
  ReadRegStr $0 HKCU "${MARKITAI_WEB_KEY}" "InstallLocation"
  StrCmp $0 "" +3
  StrCmp $0 $INSTDIR ours
  StrCmp $0 "$INSTDIR\" ours
  IfFileExists "$INSTDIR\${MARKITAI_WEB_EXE}" 0 +2
  IfFileExists "$INSTDIR\Uninstall.exe" ours
  ClearErrors
  FindFirst $0 $1 "$INSTDIR\*"
  IfErrors empty
  next:
    StrCmp $1 "" closed
    StrCmp $1 "." more
    StrCmp $1 ".." more
    FindClose $0
    Goto foreign
  more:
    FindNext $0 $1
    Goto next
  closed:
  FindClose $0
  empty:
  ours:
  Pop $1
  Pop $0
  Push 0
  Return
  foreign:
  Pop $1
  Pop $0
  Push 1
FunctionEnd
!macroend
!insertmacro MarkitaiWebForeign ""

; Quits a running Markitai Web once the user agrees, asking with the
; message on the stack; a silent run agrees. Aborts when the user does not.
; Only the app that customInstall registered in $INSTDIR is asked: a build
; from before these requests would take them for a launch, open its
; window and keep the installer waiting; mygo's own close quits that one.
!macro MarkitaiWebQuit un
Function ${un}MarkitaiWebQuit
  Exch $0
  Push $1
  Push $2
  IfFileExists "$INSTDIR\${MARKITAI_WEB_EXE}" 0 done
  ReadRegDWORD $1 HKCU "${MARKITAI_WEB_KEY}" "QuitsOnRequest"
  StrCmp $1 "1" 0 done
  ReadRegStr $1 HKCU "${MARKITAI_WEB_KEY}" "InstallLocation"
  StrCmp $1 $INSTDIR 0 done
  ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --is-running' $1
  StrCmp $1 "2" 0 done
  MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION $0 /SD IDOK IDOK quit
  Pop $2
  Pop $1
  Pop $0
  Abort
  quit:
  ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --quit-now'
  ; The app stops its server first; mygo's close after this stops what is
  ; left after 10 s.
  StrCpy $2 0
  wait:
    ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --is-running' $1
    StrCmp $1 "2" 0 done
    IntOp $2 $2 + 1
    IntCmp $2 40 done
    Sleep 250
    Goto wait
  done:
  Pop $2
  Pop $1
  Pop $0
FunctionEnd
!macroend
!insertmacro MarkitaiWebQuit ""
!insertmacro MarkitaiWebQuit "un."

!macro customInit
  ; NSIS ends the folder it reads from the uninstall entry
  ; (InstallDirRegKey) with a backslash, and the directory page of a
  ; reinstall shows the variable as it is. $INSTDIR reads without it, so
  ; assigning it to itself drops it; the page gives a drive's root its own.
  StrCpy $INSTDIR $INSTDIR
  IfSilent 0 +6
  Call MarkitaiWebForeign
  Pop $0
  StrCmp $0 "1" 0 +3
  SetErrorLevel 2
  Quit
  Push $(MARKITAI_WEB_RUNNING)
  Call MarkitaiWebQuit
!macroend

; The directory page: Next stays off for a folder that is not Markitai
; Web's and holds files, and the page's text says why; it keeps the page's
; own text, read the first time, for the other folders.
Function .onVerifyInstDir
  Push $0
  Push $1
  FindWindow $0 "#32770" "" $HWNDPARENT
  GetDlgItem $0 $0 1006
  StrCmp $MarkitaiWebIntro "" 0 +2
  System::Call 'user32::GetWindowTextW(p r0, w .r1, i ${NSIS_MAX_STRLEN})'
  StrCmp $MarkitaiWebIntro "" 0 +2
  StrCpy $MarkitaiWebIntro $1
  Call MarkitaiWebForeign
  Pop $1
  StrCmp $1 "1" foreign
  SendMessage $0 ${WM_SETTEXT} 0 "STR:$MarkitaiWebIntro"
  Pop $1
  Pop $0
  Return
  foreign:
  SendMessage $0 ${WM_SETTEXT} 0 "STR:$(MARKITAI_WEB_FOREIGN)"
  Pop $1
  Pop $0
  Abort
FunctionEnd

Function un.onInit
  Push $(MARKITAI_WEB_RUNNING_UN)
  Call un.MarkitaiWebQuit
FunctionEnd

; The app installed now answers --is-running and --quit-now
; (MarkitaiWebQuit).
!macro customInstall
  WriteRegDWORD HKCU "${MARKITAI_WEB_KEY}" "QuitsOnRequest" 1
!macroend
