// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import Foundation
import XCTest
@testable import ToshLLM

/// Continuation and regeneration start a fresh stream and must drop any stale `lastError` banner.
@MainActor
final class ChatContinueRegenerateErrorTests: XCTestCase {
    private final class RequestCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0

        func next() -> Int {
            lock.lock()
            defer { lock.unlock() }
            value += 1
            return value
        }

        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private var savedDefaults: [(String, Any?)] = []
    private var servers: [LoopbackHTTPServer] = []
    private var tempDirectories: [URL] = []

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults.standard
        let overrides: [(String, Any?)] = [
            (SettingsKeys.ctx, 16_384),
            (SettingsKeys.chatAutoCompact, false),
            (SettingsKeys.agentToolsEnabled, false),
            (SettingsKeys.sympyEnabled, false),
            (SettingsKeys.scientificEnabled, false),
            (SettingsKeys.mathAgentEnabled, false),
            (SettingsKeys.jsSandboxEnabled, false),
            (SettingsKeys.memoryToolsEnabled, false),
            (SettingsKeys.chatMapToolEnabled, false),
            (SettingsKeys.chatRoadRouteEnabled, false),
            (SettingsKeys.routerMode, false),
            (SettingsKeys.persistCache, false),
            (SettingsKeys.faAmd, true),
            (SettingsKeys.loadVision, false),
            (SettingsKeys.modelPath, nil),
            (SettingsKeys.apiKeyEnabled, false),
            (SettingsKeys.extensionsFolder,
             FileManager.default.temporaryDirectory.appendingPathComponent("tosh-no-ext-\(UUID().uuidString)").path),
            (SettingsKeys.mcpServers, nil)
        ]
        savedDefaults = overrides.map { ($0.0, defaults.object(forKey: $0.0)) }
        for (key, value) in overrides {
            if let value {
                defaults.set(value, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }
    }

    override func tearDown() {
        for server in servers {
            server.stop()
        }
        servers.removeAll()
        for directory in tempDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        tempDirectories.removeAll()
        let defaults = UserDefaults.standard
        for (key, original) in savedDefaults {
            if let original {
                defaults.set(original, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }
        savedDefaults.removeAll()
        super.tearDown()
    }

    func testContinueResponseClearsStaleLastError() async throws {
        let (store, port, counter, _) = try await makeStoreWithReplies { attempt in
            .respond(status: 200,
                     headers: ["Content-Type": "text/event-stream"],
                     body: Self.sseStream(
                        content: attempt == 1 ? "First part" : " second part",
                        finishReason: attempt == 1 ? "length" : "stop",
                        promptTokens: attempt == 1 ? 10 : 20,
                        completionTokens: attempt == 1 ? 512 : 4))
        }

        try await sendTruncatedPrompt(store: store, port: port)

        store.continueResponse(port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        XCTAssertTrue(store.generating)
        XCTAssertNil(store.lastError, "continueResponse must clear stale lastError when starting")

        try await waitUntil { !store.generating }
        XCTAssertNil(store.lastError)
        XCTAssertEqual(counter.count, 2)
        XCTAssertTrue(store.current?.messages.last?.parts.body.contains("second part") == true)
    }

    func testRegenerateClearsStaleLastError() async throws {
        let (store, port, counter, _) = try await makeStoreWithReplies { attempt in
            .respond(status: 200,
                     headers: ["Content-Type": "text/event-stream"],
                     body: Self.sseStream(
                        content: attempt == 1 ? "Truncated draft" : "Complete answer",
                        finishReason: attempt == 1 ? "length" : "stop",
                        promptTokens: 10,
                        completionTokens: attempt == 1 ? 512 : 3))
        }

        try await sendTruncatedPrompt(store: store, port: port)

        store.regenerate(port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        XCTAssertTrue(store.generating)
        XCTAssertNil(store.lastError, "regenerate must clear stale lastError when starting")

        try await waitUntil { !store.generating }
        XCTAssertNil(store.lastError)
        XCTAssertEqual(counter.count, 2)
        XCTAssertEqual(store.current?.messages.last?.parts.body, "Complete answer")
    }

    func testStallThenSendClearsAndKeepsLastErrorNil() async throws {
        let (store, port, counter, server) = try await makeStoreWithReplies { attempt in
            switch attempt {
            case 1:
                let partial = #"data: {"choices":[{"index":0,"delta":{"content":"Partial"},"finish_reason":null}]}"#
                return .stream(first: Data((partial + "\n\n").utf8))
            default:
                let first = #"data: {"choices":[{"index":0,"delta":{"content":"Recovered "},"finish_reason":null}]}"#
                return .stream(first: Data((first + "\n\n").utf8))
            }
        }

        store.send(text: "Stall request", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        try await waitUntil { store.live.visibleText == "Partial" }

        store.handleStreamStall(reason: "test stall")
        XCTAssertFalse(store.generating)
        XCTAssertEqual(store.live.visibleText, "")
        XCTAssertEqual(store.lastError, ChatStore.stallMessage(moe: false))
        XCTAssertEqual(store.current?.messages.map(\.role), ["user"],
                       "handleStreamStall must clean up the empty assistant placeholder")

        store.send(text: "Retry after stall", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        XCTAssertTrue(store.generating)
        XCTAssertEqual(store.live.visibleText, "")
        XCTAssertNil(store.lastError, "send after stall must clear stale lastError immediately")

        try await waitUntil { store.live.visibleText == "Recovered " && store.finishedStreamTaskCount >= 1 }
        XCTAssertTrue(store.generating, "unwinding stalled task must not clear generating on active stream")
        XCTAssertNil(store.lastError, "unwinding stalled task must not restore stale error while streaming")

        server.push(Self.sseStream(content: "answer", finishReason: "stop",
                                   promptTokens: 10, completionTokens: 2))
        try await waitUntil { !store.generating }
        XCTAssertNil(store.lastError)
        XCTAssertEqual(counter.count, 2)
        XCTAssertEqual(store.current?.messages.map(\.role), ["user", "user", "assistant"])
        XCTAssertEqual(store.current?.messages.last?.parts.body, "Recovered answer")
    }

    func testStopPreventsQueuedDeliveryAndStaleError() async throws {
        let (store, port, counter, server) = try await makeStoreWithReplies { _ in
            let partial = #"data: {"choices":[{"index":0,"delta":{"content":"Partial"},"finish_reason":null}]}"#
            return .stream(first: Data((partial + "\n\n").utf8))
        }

        store.send(text: "Initial question", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        try await waitUntil { store.live.visibleText == "Partial" }

        store.queueMessage(text: "Queued follow-up", attachments: [], images: [])
        store.stop()

        try await waitUntil {
            !store.generating && server.requests.contains { $0.method == "DELETE" }
        }
        XCTAssertNil(store.lastError)
        XCTAssertNil(store.queuedMessage, "stop must clear the stopped conversation's queued message banner")
        XCTAssertEqual(counter.count, 1, "stop must not launch queued follow-up request")
        XCTAssertEqual(store.current?.messages.map(\.role), ["user", "assistant"])
        XCTAssertEqual(store.current?.messages.last?.parts.body, "Partial")
    }

    func testCancelledSlotRestoreClearsSlotConvIDAndSerializesNextRestore() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let probe = SlotRestoreProbe()
        defer { probe.releaseSemaphore.signal() }
        let (store, port, _, server, slotDir) = try await makeStoreWithRequestHandler(isolateSlots: true) { request, counter in
            if let slot = Self.parseSlotAction(request) {
                if slot.action == "restore", slot.filename == probe.targetFilename {
                    probe.markEntered()
                    _ = probe.releaseSemaphore.wait(timeout: .now() + 5)
                }
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            let n = counter.next()
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Answer \(n)", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let convA = try XCTUnwrap(store.current?.id)
        try createSlotFile(for: convA, in: slotDir)
        let fileA = "\(convA.uuidString).bin"
        store.send(text: "Turn A1", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil {
            !store.generating
                && server.requests.compactMap(Self.parseSlotAction).contains { $0 == ("save", fileA) }
        }

        store.newConversation()
        let convB = try XCTUnwrap(store.current?.id)
        XCTAssertNotEqual(convB, convA)
        probe.setTargetFilename("\(convB.uuidString).bin")
        try createSlotFile(for: convB, in: slotDir)

        store.send(text: "Turn B1", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { probe.entered }

        // Stopping mid-restore cancels the chat turn while the in-flight HTTP restore finishes.
        store.stop()
        await Task.yield()
        XCTAssertTrue(store.generating, "stop must keep generating true until in-flight HTTP restore finishes")
        probe.releaseSemaphore.signal()
        try await waitUntil { !store.generating }

        XCTAssertNil(store.lastError)
        XCTAssertEqual(store.conversations.first(where: { $0.id == convB })?.messages.map(\.role), ["user"])

        // Switching back to A must re-restore A because B's cancelled restore reset slotConvID.
        store.currentID = convA
        store.send(text: "Turn A2", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil {
            !store.generating
                && server.requests.compactMap(Self.parseSlotAction).filter { $0 == ("save", fileA) }.count == 2
        }

        let slotActions = server.requests.compactMap(Self.parseSlotAction).map { "\($0.action):\($0.filename)" }
        let fileB = "\(convB.uuidString).bin"
        XCTAssertEqual(slotActions, [
            "restore:\(fileA)",
            "save:\(fileA)",
            "restore:\(fileB)",
            "restore:\(fileA)",
            "save:\(fileA)"
        ])
        let completions = server.requests.filter { $0.path == "/v1/chat/completions" }
        XCTAssertEqual(completions.count, 2, "cancelled prepareSlot must abort before issuing completion request")
    }

    func testFailedSlotRestoreResetsSlotConvID() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let convBHolder = SlotRestoreProbe()
        let (store, port, _, server, slotDir) = try await makeStoreWithRequestHandler(isolateSlots: true) { request, counter in
            if let slot = Self.parseSlotAction(request) {
                if slot.action == "restore", slot.filename == convBHolder.targetFilename {
                    return .respond(status: 500, headers: [:], body: Data())
                }
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            let n = counter.next()
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Reply \(n)", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let convA = try XCTUnwrap(store.current?.id)
        try createSlotFile(for: convA, in: slotDir)
        store.send(text: "Turn A1", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { !store.generating }

        store.newConversation()
        let convB = try XCTUnwrap(store.current?.id)
        XCTAssertNotEqual(convB, convA)
        convBHolder.setTargetFilename("\(convB.uuidString).bin")
        try createSlotFile(for: convB, in: slotDir)

        store.send(text: "Turn B1", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { !store.generating }

        // A second turn in B must retry restore because the 500 response left slotConvID unset.
        store.send(text: "Turn B2", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { !store.generating }

        // Returning to A after a failed restore in B must restore A again, not assume A stayed resident.
        store.currentID = convA
        store.send(text: "Turn A2", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { !store.generating }

        let fileA = "\(convA.uuidString).bin"
        let fileB = "\(convB.uuidString).bin"
        let restores = server.requests.compactMap(Self.parseSlotAction).filter { $0.action == "restore" }.map(\.filename)
        XCTAssertEqual(restores, [fileA, fileB, fileB, fileA])
    }

    func testDeletingBackgroundGeneratingConversationStopsStreamAndSkipsSave() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let (store, port, _, server, slotDir) = try await makeStoreWithRequestHandler(isolateSlots: true) { request, _ in
            if Self.parseSlotAction(request) != nil {
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            let partial = #"data: {"choices":[{"index":0,"delta":{"content":"Partial"},"finish_reason":null}]}"#
            return .stream(first: Data((partial + "\n\n").utf8))
        }

        let convA = try XCTUnwrap(store.current?.id)
        store.send(text: "Stream in A", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { store.live.visibleText == "Partial" }

        // Switch to another conversation while convA keeps streaming, then delete convA.
        store.newConversation()
        let convB = try XCTUnwrap(store.current?.id)
        XCTAssertNotEqual(convA, convB)
        let convAObj = try XCTUnwrap(store.conversations.first { $0.id == convA })
        store.delete(convAObj)

        try await waitUntil { !store.generating }
        XCTAssertNil(store.lastError)
        XCTAssertEqual(store.currentID, convB)
        let fileA = "\(convA.uuidString).bin"
        XCTAssertFalse(server.requests.compactMap(Self.parseSlotAction).contains { $0 == ("save", fileA) })
        XCTAssertFalse(FileManager.default.fileExists(atPath: slotDir.appendingPathComponent(fileA).path))
    }

    func testDeletingIdleCurrentConversationKeepsBackgroundStreamAlive() async throws {
        let (store, port, _, server) = try await makeStoreWithReplies { _ in
            let partial = #"data: {"choices":[{"index":0,"delta":{"content":"Background "},"finish_reason":null}]}"#
            return .stream(first: Data((partial + "\n\n").utf8))
        }

        let convA = try XCTUnwrap(store.current?.id)
        store.send(text: "Stream in A", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { store.live.visibleText == "Background " }

        // Switch to an idle conversation B, then delete B: the unrelated stream in A must keep running.
        store.newConversation()
        let convB = try XCTUnwrap(store.current?.id)
        XCTAssertNotEqual(convA, convB)
        let convBObj = try XCTUnwrap(store.conversations.first { $0.id == convB })
        store.delete(convBObj)

        XCTAssertTrue(store.generating, "deleting an idle conversation must not cancel another conversation's stream")
        XCTAssertEqual(store.generatingConvID, convA)

        server.push(Self.sseStream(content: "answer", finishReason: "stop",
                                   promptTokens: 10, completionTokens: 2))
        try await waitUntil { !store.generating }
        XCTAssertNil(store.lastError)
        XCTAssertEqual(store.conversations.first(where: { $0.id == convA })?.messages.last?.parts.body,
                       "Background answer")
    }

    func testDeletingConversationDuringInFlightSaveRemovesOrphanSlotFile() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let customSlotDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-chat-slot-\(UUID().uuidString)")
        tempDirectories.append(customSlotDir)
        let probe = SlotRestoreProbe()
        defer { probe.releaseSemaphore.signal() }
        let (store, port, _, _, _) = try await makeStoreWithRequestHandler(isolateSlots: true,
                                                                           slotDirectory: customSlotDir) { request, _ in
            if let slot = Self.parseSlotAction(request), slot.action == "save" {
                probe.setTargetFilename(slot.filename)
                probe.markEntered()
                _ = probe.releaseSemaphore.wait(timeout: .now() + 5)
                try? FileManager.default.createDirectory(at: customSlotDir, withIntermediateDirectories: true)
                try? Data("orphan".utf8).write(to: customSlotDir.appendingPathComponent(slot.filename))
                probe.markFinished()
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Saved reply", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let convA = try XCTUnwrap(store.current?.id)
        store.send(text: "Turn A", port: port, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { probe.entered }

        let convAObj = try XCTUnwrap(store.conversations.first { $0.id == convA })
        store.delete(convAObj)
        probe.releaseSemaphore.signal()

        let fileA = customSlotDir.appendingPathComponent("\(convA.uuidString).bin")
        try await waitUntil { !store.generating && probe.finished }
        try await waitUntil { !FileManager.default.fileExists(atPath: fileA.path) }
    }

    func testStopDoesNotCancelInFlightPostTurnSave() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let customSlotDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-chat-slot-\(UUID().uuidString)")
        tempDirectories.append(customSlotDir)
        let probe = SlotRestoreProbe()
        defer { probe.releaseSemaphore.signal() }
        let (store, port1, _, _, _) = try await makeStoreWithRequestHandler(isolateSlots: true,
                                                                           slotDirectory: customSlotDir) { request, _ in
            if let slot = Self.parseSlotAction(request), slot.action == "save" {
                probe.markEntered()
                _ = probe.releaseSemaphore.wait(timeout: .now() + 5)
                probe.markFinished()
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "A reply", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let server2 = try LoopbackHTTPServer()
        servers.append(server2)
        let port2 = try await server2.start { request in
            if Self.parseSlotAction(request) != nil {
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "B reply", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let convA = try XCTUnwrap(store.current?.id)
        store.send(text: "Turn A", port: port1, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { probe.entered }

        // Calling stop() while post-turn save is in flight must preserve serialization on the live server.
        store.stop()
        store.newConversation()
        let convB = try XCTUnwrap(store.current?.id)
        XCTAssertNotEqual(convA, convB)
        try createSlotFile(for: convB, in: customSlotDir)

        store.send(text: "Turn B", port: port2, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { store.awaitingSlotPreparation }
        XCTAssertTrue(server2.requests.isEmpty,
                      "Turn B must wait for Turn A's in-flight save even if stop() ran in between")

        probe.releaseSemaphore.signal()
        try await waitUntil { !store.generating && probe.finished }
        let fileB = "\(convB.uuidString).bin"
        XCTAssertTrue(server2.requests.compactMap(Self.parseSlotAction).contains { $0 == ("restore", fileB) })
    }

    func testStallCancelsPriorTurnInFlightSaveTask() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.persistCache)
        let customSlotDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-chat-slot-\(UUID().uuidString)")
        tempDirectories.append(customSlotDir)
        let probe = SlotRestoreProbe()
        defer { probe.releaseSemaphore.signal() }
        let (store, port1, _, _, _) = try await makeStoreWithRequestHandler(isolateSlots: true,
                                                                           slotDirectory: customSlotDir) { request, _ in
            if let slot = Self.parseSlotAction(request), slot.action == "save" {
                probe.markEntered()
                _ = probe.releaseSemaphore.wait(timeout: .now() + 5)
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "A reply", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        let server2 = try LoopbackHTTPServer()
        servers.append(server2)
        let port2 = try await server2.start { request in
            if Self.parseSlotAction(request) != nil {
                return .respond(status: 200, headers: [:], body: Data("{}".utf8))
            }
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Recovered", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        store.send(text: "Turn A", port: port1, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { probe.entered }

        // Start Turn B on port1 (blocked in prepareSlot awaiting Turn A's save), then trigger stall.
        store.newConversation()
        store.send(text: "Stalled B", port: port1, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil { store.awaitingSlotPreparation }
        store.handleStreamStall(reason: "stall while prior save in flight")
        XCTAssertFalse(store.generating)

        // Next turn on port2 must proceed immediately without waiting for port1's cancelled save semaphore.
        store.send(text: "Fresh C", port: port2, temperature: 0.7, maxTokens: 512, system: "", thinking: false)
        try await waitUntil({ !store.generating }, timeout: 2)
        XCTAssertNil(store.lastError)
        XCTAssertEqual(store.current?.messages.last?.parts.body, "Recovered")
    }

    func testStopDuringToolDiscoverySkipsCompletionRequest() async throws {
        UserDefaults.standard.set(true, forKey: SettingsKeys.sympyEnabled)
        let probe = SlotRestoreProbe()
        defer { probe.releaseSemaphore.signal() }
        let (store, port, counter, server, _) = try await makeStoreWithRequestHandler { request, counter in
            if request.path == "/tools" {
                probe.markEntered()
                _ = probe.releaseSemaphore.wait(timeout: .now() + 5)
                return .respond(status: 200, headers: ["Content-Type": "application/json"], body: Data("[]".utf8))
            }
            let n = counter.next()
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Unexpected \(n)", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }

        store.send(text: "Stop before completions", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        try await waitUntil { probe.entered }

        store.stop()
        probe.releaseSemaphore.signal()
        try await waitUntil { !store.generating && store.finishedStreamTaskCount == 1 }

        XCTAssertNil(store.lastError)
        XCTAssertEqual(counter.count, 0, "stopped turn must not issue a completion request")
        XCTAssertEqual(server.requests.filter { $0.path == "/tools" }.count, 1)
        XCTAssertFalse(server.requests.contains { $0.path == "/v1/chat/completions" },
                       "cancelled task must not issue POST /v1/chat/completions after tool discovery")
        XCTAssertEqual(store.current?.messages.map(\.role), ["user"])
    }

    func testStopBetweenCheckCancellationAndStreamLaunchSkipsPostAndDelete() async throws {
        let hook = AsyncHookBox()
        let (store, port, counter, server, _) = try await makeStoreWithRequestHandler(
            onBeforeStreamRequest: { await hook.run() }
        ) { _, counter in
            let n = counter.next()
            return .respond(status: 200,
                            headers: ["Content-Type": "text/event-stream"],
                            body: Self.sseStream(content: "Unexpected \(n)", finishReason: "stop",
                                                 promptTokens: 10, completionTokens: 2))
        }
        hook.set { @MainActor [weak store] in store?.stop() }

        store.send(text: "Boundary stop", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        try await waitUntil { !store.generating && store.finishedStreamTaskCount == 1 }

        XCTAssertNil(store.lastError)
        XCTAssertEqual(counter.count, 0)
        XCTAssertTrue(server.requests.isEmpty,
                      "stop right before bytes(for:delegate:) must neither send POST nor fire premature DELETE")
        XCTAssertEqual(store.current?.messages.map(\.role), ["user"])
    }

    private final class AsyncHookBox: @unchecked Sendable {
        private let lock = NSLock()
        private var handler: (@Sendable () async -> Void)?

        func set(_ handler: @escaping @Sendable () async -> Void) {
            lock.withLock { self.handler = handler }
        }

        func run() async {
            let fn = lock.withLock { handler }
            await fn?()
        }
    }

    private final class SlotRestoreProbe: @unchecked Sendable {
        private let lock = NSLock()
        private var target: String = ""
        private var didEnter = false
        private var didFinish = false
        let releaseSemaphore = DispatchSemaphore(value: 0)

        var targetFilename: String { lock.withLock { target } }
        var entered: Bool { lock.withLock { didEnter } }
        var finished: Bool { lock.withLock { didFinish } }

        func setTargetFilename(_ name: String) { lock.withLock { target = name } }
        func markEntered() { lock.withLock { didEnter = true } }
        func markFinished() { lock.withLock { didFinish = true } }
    }

    private nonisolated static func parseSlotAction(_ request: LoopbackHTTPServer.Request) -> (action: String, filename: String)? {
        guard request.method == "POST",
              request.path.hasPrefix("/slots/0?"),
              let comps = URLComponents(string: "http://127.0.0.1" + request.path),
              let action = comps.queryItems?.first(where: { $0.name == "action" })?.value,
              let filename = request.json?["filename"] as? String else { return nil }
        return (action, filename)
    }

    private func createSlotFile(for convID: UUID, in directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("\(convID.uuidString).bin")
        try Data("slot".utf8).write(to: file)
    }

    private func makeStoreWithReplies(
        _ respond: @escaping @Sendable (Int) -> LoopbackHTTPServer.Reply
    ) async throws -> (ChatStore, Int, RequestCounter, LoopbackHTTPServer) {
        let (store, port, counter, server, _) = try await makeStoreWithRequestHandler { _, counter in
            respond(counter.next())
        }
        return (store, port, counter, server)
    }

    private func makeStoreWithRequestHandler(
        isolateSlots: Bool = false,
        slotDirectory: URL? = nil,
        onBeforeStreamRequest: (@Sendable () async -> Void)? = nil,
        _ respond: @escaping @Sendable (LoopbackHTTPServer.Request, RequestCounter) -> LoopbackHTTPServer.Reply
    ) async throws -> (ChatStore, Int, RequestCounter, LoopbackHTTPServer, URL) {
        let counter = RequestCounter()
        let server = try LoopbackHTTPServer()
        servers.append(server)
        let port = try await server.start { request in
            if request.method == "DELETE" {
                return .respond(status: 204, headers: [:], body: Data())
            }
            return respond(request, counter)
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tosh-chat-error-\(UUID().uuidString)")
        tempDirectories.append(directory)
        let slotDir = slotDirectory ?? directory.appendingPathComponent("slots")
        let store = ChatStore(storageDirectory: directory,
                              slotCacheDirectory: isolateSlots ? slotDir : nil,
                              stopStalledEngine: { _ in },
                              onBeforeStreamRequest: onBeforeStreamRequest)
        store.newConversation()
        return (store, Int(port), counter, server, slotDir)
    }

    private func sendTruncatedPrompt(store: ChatStore, port: Int) async throws {
        store.send(text: "Write an answer", port: port, temperature: 0.7, maxTokens: 512,
                   system: "", thinking: false)
        try await waitUntil { !store.generating }
        XCTAssertEqual(store.lastError, ChatStore.cutAnswerMessage())
    }

    private struct WaitTimeoutError: Error {}

    private func waitUntil(_ condition: () -> Bool, timeout: TimeInterval = 5) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() >= deadline {
                XCTFail("Timed out waiting for condition")
                throw WaitTimeoutError()
            }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    private nonisolated static func sseStream(content: String, finishReason: String,
                                              promptTokens: Int, completionTokens: Int) -> Data {
        let chunks = [
            #"data: {"choices":[{"index":0,"delta":{"content":"\#(content)"},"finish_reason":null}]}"#,
            #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"\#(finishReason)"}]}"#,
            #"data: {"choices":[],"usage":{"prompt_tokens":\#(promptTokens),"completion_tokens":\#(completionTokens)}}"#,
            "data: [DONE]"
        ]
        return Data((chunks.joined(separator: "\n\n") + "\n\n").utf8)
    }
}
