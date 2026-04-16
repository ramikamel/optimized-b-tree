#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "adaptive_btree/node.hpp"

namespace abt
{

    struct InnerEntry
    {
        std::string key;
        NodeId right_child;
    };

    struct InnerMaterialized
    {
        NodeId left_child;
        std::vector<InnerEntry> entries;
    };

    class InnerNode final : public Node
    {
    public:
        explicit InnerNode(NodeId id);

        InnerMaterialized materialize() const;
        bool rebuild(NodeId left_child, const std::vector<InnerEntry> &sorted_entries);

        std::size_t childIndexForKey(std::string_view key) const;
        NodeId childAt(std::size_t child_index) const;
        NodeId childForKey(std::string_view key) const;

    private:
        int compareKeyAt(std::uint16_t slot_index, std::string_view key) const;
        static std::string commonPrefix(const std::vector<InnerEntry> &sorted_entries);
    };

} // namespace abt
